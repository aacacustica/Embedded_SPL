#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#include <stdexcept>
#include <yaml-cpp/yaml.h>

float PREF = 2e-6f;

constexpr int NUM_BANDS = 25;



struct SosBank {

    //dimensiones y fs
    int nbands = 0;
    int nsec = 0;
    int fs = 0;

    //coeficientes
    std::vector<float> b0,b1,b2;
    std::vector<float> a1,a2;

    //estados
    std::vector<float> z1,z2;

    //indice para acceder a los coeficientes y estados
    inline int idx(int b, int s) const {
        return b * nsec + s;
    }

    void resize(int bands, int sections) {
        nbands = bands;
        nsec = sections;

        int total = bands * sections;

        b0.resize(total);
        b1.resize(total);
        b2.resize(total);
        a1.resize(total);
        a2.resize(total);

        z1.resize(total, 0.0f);
        z2.resize(total, 0.0f);
    }

    void load_coefficients(const std::string& path, int expected_fs){
        YAML::Node root = YAML::LoadFile(path);

        fs = root["fs"].as<int>();
        if (fs != expected_fs){
            throw std::runtime_error("Sample rate in SOS file does not match expected sample rate.");
        }

        const auto& sos_yaml = root["sos_bank"];

        int bands = sos_yaml.size();
        int sections = sos_yaml[0].size();

        resize(bands, sections);

        for (int b = 0; b < bands; b++){
            for (int s = 0; s < sections; s++){

                auto sec = sos_yaml[b][s];
                int i = idx(b, s);

                b0[i] = sec[0].as<float>();
                b1[i] = sec[1].as<float>();
                b2[i] = sec[2].as<float>();

                a1[i] = sec[4].as<float>();
                a2[i] = sec[5].as<float>();
            }
        }

    }

    void process_sample(float x, std::vector<float>& y_band)
    {
        for (int b = 0; b < nbands; b++)
        {
            float y = x;

            for (int s = 0; s < nsec; s++)
            {
                int i = idx(b,s);

                float out = b0[i] * y + z1[i];

                z1[i] = b1[i]*y - a1[i]*out + z2[i];
                z2[i] = b2[i]*y - a2[i]*out;

                y = out;
            }

            y_band[b] = y;
        }
    }
};

struct FrameFeatures {
    std::chrono::system_clock::time_point timestamp;

    float energyA = 0.0f;
    float energyC = 0.0f;
    float energyZ = 0.0f;

    std::vector<float> energyBands;
};


struct Result {
    float LA, LC, LZ;
    float Lmax, Lmin;
    std::vector<float> bands;
};

struct FilterState {
    std::vector<float> stateA;
    std::vector<float> stateC;
};

struct IIRFilter {
    float b0, b1, b2;
    float a1, a2;
    
    float z1 = 0.0f;
    float z2 = 0.0f;

    inline float step(float x) {
        float out = b0*x + z1;

        z1 = b1*x - a1*out + z2;
        z2 = b2*x - a2*out;

        return out;
    }
};


struct SystemDSP {

    SosBank bank;

    IIRFilter filterA;
    IIRFilter filterC;

    std::vector<float> y_band;

    int nbands;

    void init(int bands) {
        nbands = bands;
        y_band.resize(bands);
    }
};


/*

Functiones auxiliares

*/

std::pair<std::vector<std::chrono::system_clock::time_point> , std::vector<int>> get_timestamps(const std::vector<float>& x,
               const std::string& audio_file,
               int window_size,
               float fs){

                //1. Quitar extensión
                std::string name_split = audio_file.substr(0, audio_file.find_last_of('.'));

                //2. Parsear fecha
                std::tm tm = {};
                std::istringstream ss(name_split);
                ss >> std::get_time(&tm, "%Y%m%d_%H%M%S");

                auto start_timestamp = std::chrono::system_clock::from_time_t(std::mktime(&tm));

                //3. Generar frame starts
                std::vector<int> frame_starts;
                for(int i = 0; i <= (int)x.size() - window_size; i += window_size){
                    frame_starts.push_back(i);
                }

                //4. Generar timestamps
                std::vector<std::chrono::system_clock::time_point> timestamps;

                for (int fstart : frame_starts){
                    double seconds = fstart / fs;
                    auto delta = std::chrono::duration<double>(seconds);
                    timestamps.push_back(start_timestamp  + std::chrono::duration_cast<std::chrono::system_clock::duration>(delta));
                }
                return {timestamps, frame_starts};
}

float get_level_db(const std::vector<float>& x, float C)
{
    /*
    The get_level_db function calculates the decibel (dB) level of a given numerical array (nc::NdArray<float>) by computing its mean square value,
    normalizing it with a predefined constant (PREF), and applying a logarithmic transformation.
    An additional constant C is added to the result to allow for adjustments.
    */

    float ms = 0.0f;
    for (auto v : x)
        ms += v * v;

    ms = ms / static_cast<float>(x.size()) + PREF;

    return 10.0f * std::log10(ms / (PREF * PREF)) + C;
}

std::vector<FrameFeatures> process_audio(
    const std::vector<float>& x,
    const std::string& audio_file,
    SystemDSP& dsp,
    int window_size,
    float fs)
{

    const float* xptr = x.data();
    auto [timestamps, frame_starts] = get_timestamps(x, audio_file, window_size, fs);
    int nbands = dsp.nbands;
    std::vector<FrameFeatures> result;
    result.reserve(frame_starts.size());

    
    

    for(size_t f = 0; f < frame_starts.size(); f++){

        int start = frame_starts[f];
        int end = std::min(start + window_size, (int)x.size());
        FrameFeatures feat;
        feat.energyBands.resize(dsp.nbands);
        feat.timestamp = timestamps[f];
        
        feat.energyBands.assign(nbands, 0.0f);

        float& eA = feat.energyA;
        float& eC = feat.energyC;
        float& eZ = feat.energyZ;

        float* eB = feat.energyBands.data();
        /* Versión sample -> todo*/

        for (int n = start; n < end; ++n)
        {
            float sample = xptr[n];

            float yA = dsp.filterA.step(sample);
            float yC = dsp.filterC.step(sample);

            dsp.bank.process_sample(sample, dsp.y_band);

            eA += yA * yA;
            eC += yC * yC;
            eZ += sample * sample;

            float* yb = dsp.y_band.data();

            for (int b = 0; b < nbands; ++b)
                eB[b] += yb[b] * yb[b];
        }

        result.push_back(std::move(feat));
    }
        /* Versión sample -> 25 bands
        for (int n = start; n < end; ++n){

            // Z (raw)
            float sample = x[n];

            //filtros A/C
            float yA = dsp.filterA.step(sample);
            float yC = dsp.filterC.step(sample);

            //SOS bank
            dsp.bank.process_sample(sample, dsp.y_band);

            //Energía A,C,Z
            feat.energyA += yA * yA;
            feat.energyC += yC * yC;
            feat.energyZ += sample * sample;

            for (int b = 0; b < dsp.nbands; ++b){
                feat.energyBands[b] += y_band[b] * y_band[b];
            }
        }
        */
        
        return result;
    }
    






std::tuple<std::vector<float>, std::vector<float>> lfilter_np(const std::vector<float>& b, const std::vector<float>& a, const std::vector<float>& x, const std::vector<float>& zi){


    std::size_t n = 0;
    std::vector<float> y = std::vector<float>();
    float a0 = 0;
    auto aa = a;
    auto bb = b; 
    std::vector<float> z;

    if (x.shape().size() != 1){
        throw std::invalid_argument("Input x must be a 1D array. ");
    }
        
    if (a.size() == 0 || b.size() == 0){
        throw std::invalid_argument("Filter coefficients a and b must be non-empty. ");
    }

    a0 = a[0];
    if (a0 ==  0){
        throw std::invalid_argument("First coefficient of 'a' must be non-zero. ");
    }
    if (a0 != 1.0){
        bb = nc::divide(b, static_cast<float>(a0));
        aa = nc::divide(a, static_cast<float>(a0));
    }
    n = std::max(aa.size(), bb.size()) - 1;

    if (zi.size() == 0) {
        z = nc::zeros<float>(n);
    } else {
        if (zi.size() != n) {
            throw std::invalid_argument("Initial state zi has incorrect length.");
        }
        z = zi;
    }

    y = nc::zeros<float>(x.size());

    float xi = 0;
    float y0 = 0;
    float bk = 0;
    float ak = 0;
    float bn = 0;
    float an = 0;

    for (int i = 0; i< x.size(); i++){
        
        xi = x[i];
        if (n == 0){
            y[i] = bb[0] * xi;
            continue;
        }
        y0 = bb[0] * xi + z[0];
        y[i] = y0;

        for (int k = 1; k < n; k++){
            bk = (k < bb.size()) ? bb[k] : 0.0f;
            ak = (k < aa.size()) ? aa[k] : 0.0f;
            z[k-1] = z[k] + bk * xi - ak * y0;
        }

        bn = (n < bb.size()) ? bb[n] : 0.0;
        an = (n < aa.size()) ? aa[n] : 0.0;
        z[n-1] = bn * xi - an * y0;
    }

    return {y, z};


}


std::tuple<std::vector<float>, std::vector<float>> sosfilt_np(const std::vector<float>& sos, const std::vector<float>& x, const std::vector<float>& zi){

    int nsec = sos.numRows();
    std::vector<float> y = x;
    std::vector<float> z = std::vector<float>();

    if (zi.size() == 0) {
        z = std::vector<float>(nsec * 2);
    } else {

        if (zi.shape().size() != 2 || zi.numRows() != nsec || zi.numCols() != 2) {
            throw std::invalid_argument("Initial state zi must have shape (nsec, 2).");
        }
        z = zi;
    }

    if (sos.shape().size() != 2 || sos.numCols() != 6) {
    throw std::invalid_argument("sos must have shape (nsec, 6).");
    }

    float b0=0,b1=0,b2=0,a0=0,a1=0,a2=0;
    float z1=0,z2=0;

    for(int s = 0; s< nsec; s++){
        
        b0 = sos(s,0);
        b1 = sos(s,1);
        b2 = sos(s,2);
        a0 = sos(s,3);
        a1 = sos(s,4);
        a2 = sos(s,5);

        if (a0 != 1.0){
            b0 = b0 / a0;
            b1 = b1 / a0;
            b2 = b2 / a0;
            a1 = a1 / a0;
            a2 = a2 / a0;
        }

        z1 = z(s,0);
        z2 = z(s,1);

        float xn = 0;
        float yn = 0;

        std::vector<float> out = std::vector<float>(y.size());

        for ( int n = 0; n < y.size(); n++){
            xn = y[n];
            yn = b0*xn + z1;
            z1 = b1*xn - a1*yn + z2;
            z2 = b2*xn - a2*yn;
            out[n] = yn;
        }

        y = out;
        z(s,0) =  static_cast<float>(z1);
        z(s,1) =  static_cast<float>(z2);
    }

    return {y,z}; 
}