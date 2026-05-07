#include <iostream>
#include <NumCpp.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>
float PREF = 2e-6f;



class sos_filter_bank {
    std::vector<float> sos;
    std::vector<float> zi;
};
s


float get_level_db(const nc::NdArray<float>& x, float C)
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

std::vector<float> process_sos_bank(frame){

}

std::tuple<std::vector<float>, std::vector<std::string>> process_audio_block(const std::vector<float>& x, const std::string& config, const std::string& state)
{
    std::vector<float> output = x;
    std::vector<std::string> metadata;
    return {output, metadata};
}


std::tuple<nc::NdArray<float>, nc::NdArray<float>> lfilter_np(const std::vector<float>& b, const std::vector<float>& a, const std::vector<float>& x, const std::vector<float>& zi){


    std::size_t n = 0;
    nc::NdArray<float> y = nc::NdArray<float>();
    double a0 = 0;
    auto aa = a;
    auto bb = b; 
    nc::NdArray<float> z;

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

    double xi = 0;
    double y0 = 0;
    double bk = 0;
    double ak = 0;
    double bn = 0;
    double an = 0;

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

std::tuple<std::vector<float>, std::vector<float>> sosfilt_np(const nc::NdArray<float>& sos, const nc::NdArray<float>& x, const nc::NdArray<float>& zi){

    int nsec = sos.numRows();
    nc::NdArray<float> y = x;
    nc::NdArray<float> z = nc::NdArray<float>();

    if (zi.size() == 0) {
        z = nc::zeros<float>(nsec, 2);
    } else {

        if (zi.shape().size() != 2 || zi.numRows() != nsec || zi.numCols() != 2) {
            throw std::invalid_argument("Initial state zi must have shape (nsec, 2).");
        }
        z = zi;
    }

    if (sos.shape().size() != 2 || sos.numCols() != 6) {
    throw std::invalid_argument("sos must have shape (nsec, 6).");
    }

    double b0=0,b1=0,b2=0,a0=0,a1=0,a2=0;
    double z1=0,z2=0;

    for(int s =0; s< nsec; s++){
        
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

        double xn = 0;
        double yn = 0;

        nc::NdArray<float> out = nc::zeros<float>(y.size());

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

std::tuple<nc::NdArray<float>, nc::NdArray<float>> sosfilt_np(const std::vector<float>& sos, const std::vector<float>& x, const std::vector<float>& zi){

    int nsec = sos.numRows();
    nc::NdArray<float> y = x;
    nc::NdArray<float> z = nc::NdArray<float>();

    if (zi.size() == 0) {
        z = nc::zeros<float>(nsec, 2);
    } else {

        if (zi.shape().size() != 2 || zi.numRows() != nsec || zi.numCols() != 2) {
            throw std::invalid_argument("Initial state zi must have shape (nsec, 2).");
        }
        z = zi;
    }

    if (sos.shape().size() != 2 || sos.numCols() != 6) {
    throw std::invalid_argument("sos must have shape (nsec, 6).");
    }

    double b0=0,b1=0,b2=0,a0=0,a1=0,a2=0;
    double z1=0,z2=0;

    for(int s =0; s< nsec; s++){
        
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

        double xn = 0;
        double yn = 0;

        nc::NdArray<float> out = nc::zeros<float>(y.size());

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