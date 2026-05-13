#include "acoustic_processor.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


// ============================================================
// Helpers
// ============================================================

static inline float level_from_sum_sq(double sum_sq, int n, float C)
{
    constexpr double P_REF = 20e-6;
    constexpr double EPS_MS = 1e-30;

    /*
        Fórmula SPL corregida:

            L = 10 * log10(mean_square / P_REF^2) + C

        con:

            P_REF = 20 µPa = 20e-6 Pa

        La versión anterior usaba PREF = 2e-6 y sumaba PREF a la media cuadrática.
        Esta versión es más coherente físicamente.
    */

    if (n <= 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    double mean_square = sum_sq / static_cast<double>(n);

    if (mean_square < EPS_MS) {
        mean_square = EPS_MS;
    }

    const double level =
        10.0 * std::log10(mean_square / (P_REF * P_REF))
        + static_cast<double>(C);

    return static_cast<float>(level);
}

// ============================================================
// IIRFilterDF2T
// ============================================================

void IIRFilterDF2T::init_from_ba(const std::vector<float>& b_in,
                                 const std::vector<float>& a_in)
{
    if (b_in.empty() || a_in.empty()) {
        throw std::runtime_error("IIR coefficients b/a cannot be empty");
    }

    const double a0 = static_cast<double>(a_in[0]);

    if (a0 == 0.0) {
        throw std::runtime_error("IIR coefficient a[0] cannot be zero");
    }

    b.resize(b_in.size());
    a.resize(a_in.size());

    for (std::size_t i = 0; i < b_in.size(); ++i) {
        b[i] = static_cast<double>(b_in[i]) / a0;
    }

    for (std::size_t i = 0; i < a_in.size(); ++i) {
        a[i] = static_cast<double>(a_in[i]) / a0;
    }

    const std::size_t nstate = std::max(b.size(), a.size()) - 1;

    z.assign(nstate, 0.0);
}


float IIRFilterDF2T::step(float x)
{
    const double xd = static_cast<double>(x);

    if (z.empty()) {
        return static_cast<float>(b[0] * xd);
    }

    const std::size_t n = z.size();

    const double y = b[0] * xd + z[0];

    for (std::size_t k = 1; k < n; ++k) {
        const double bk = k < b.size() ? b[k] : 0.0;
        const double ak = k < a.size() ? a[k] : 0.0;

        z[k - 1] = z[k] + bk * xd - ak * y;
    }

    const double bn = n < b.size() ? b[n] : 0.0;
    const double an = n < a.size() ? a[n] : 0.0;

    z[n - 1] = bn * xd - an * y;

    return static_cast<float>(y);
}


void IIRFilterDF2T::reset()
{
    std::fill(z.begin(), z.end(), 0.0);
}

// ============================================================
// Biquad
// ============================================================

void Biquad::init(double raw_b0,
                  double raw_b1,
                  double raw_b2,
                  double raw_a0,
                  double raw_a1,
                  double raw_a2)
{
    if (raw_a0 == 0.0) {
        throw std::runtime_error("Biquad a0 cannot be zero");
    }

    b0 = raw_b0 / raw_a0;
    b1 = raw_b1 / raw_a0;
    b2 = raw_b2 / raw_a0;
    a1 = raw_a1 / raw_a0;
    a2 = raw_a2 / raw_a0;

    reset();
}


float Biquad::step(float x)
{
    const double xd = static_cast<double>(x);

    const double y = b0 * xd + z1;

    z1 = b1 * xd - a1 * y + z2;
    z2 = b2 * xd - a2 * y;

    return static_cast<float>(y);
}


void Biquad::reset()
{
    z1 = 0.0;
    z2 = 0.0;
}


// ============================================================
// BiquadCascade
// ============================================================

void BiquadCascade::init_from_ba(const std::vector<float>& b,
                                 const std::vector<float>& a)
{
    if (b.size() != 3 || a.size() != 3) {
        throw std::runtime_error(
            "Expected b/a filters with exactly 3 coefficients each"
        );
    }

    sections.resize(1);

    sections[0].init(
        b[0], b[1], b[2],
        a[0], a[1], a[2]
    );
}


void BiquadCascade::init_from_sos_flat(const std::vector<float>& sos_flat,
                                       int n_sections)
{
    if (n_sections <= 0) {
        throw std::runtime_error("n_sections must be positive");
    }

    if (static_cast<int>(sos_flat.size()) != n_sections * 6) {
        throw std::runtime_error("Invalid SOS flat size");
    }

    sections.resize(n_sections);

    for (int s = 0; s < n_sections; ++s) {
        const int i = s * 6;

        sections[s].init(
            sos_flat[i + 0],
            sos_flat[i + 1],
            sos_flat[i + 2],
            sos_flat[i + 3],
            sos_flat[i + 4],
            sos_flat[i + 5]
        );
    }
}


float BiquadCascade::step(float x)
{
    float y = x;

    for (auto& sec : sections) {
        y = sec.step(y);
    }

    return y;
}


void BiquadCascade::reset()
{
    for (auto& sec : sections) {
        sec.reset();
    }
}


// ============================================================
// SosBank
// ============================================================

int SosBank::idx(int band, int section) const
{
    return band * nsec + section;
}


void SosBank::init_from_flat_sos(const float* data,
                                 int bands,
                                 int sections)
{
    if (data == nullptr) {
        throw std::runtime_error("SOS data pointer is null");
    }

    if (bands <= 0) {
        throw std::runtime_error("Number of bands must be positive");
    }

    if (sections <= 0) {
        throw std::runtime_error("Number of SOS sections must be positive");
    }

    nbands = bands;
    nsec = sections;

    const int total = nbands * nsec;

    b0.resize(total);
    b1.resize(total);
    b2.resize(total);
    a1.resize(total);
    a2.resize(total);

    z1.assign(total, 0.0);
    z2.assign(total, 0.0);

    for (int b = 0; b < nbands; ++b) {
        for (int s = 0; s < nsec; ++s) {
            const int out_i = idx(b, s);
            const int in_i = ((b * nsec) + s) * 6;

            const double raw_b0 = data[in_i + 0];
            const double raw_b1 = data[in_i + 1];
            const double raw_b2 = data[in_i + 2];
            const double raw_a0 = data[in_i + 3];
            const double raw_a1 = data[in_i + 4];
            const double raw_a2 = data[in_i + 5];

            if (raw_a0 == 0.0) {
                throw std::runtime_error("SOS a0 cannot be zero");
            }

            b0[out_i] = raw_b0 / raw_a0;
            b1[out_i] = raw_b1 / raw_a0;
            b2[out_i] = raw_b2 / raw_a0;
            a1[out_i] = raw_a1 / raw_a0;
            a2[out_i] = raw_a2 / raw_a0;
        }
    }
}


void SosBank::process_sample(float x, float* y_band)
{
    if (y_band == nullptr) {
        throw std::runtime_error("y_band pointer is null");
    }

    for (int b = 0; b < nbands; ++b) {
        double y = static_cast<double>(x);

        const int base = b * nsec;

        for (int s = 0; s < nsec; ++s) {
            const int i = base + s;

            const double out = b0[i] * y + z1[i];

            z1[i] = b1[i] * y - a1[i] * out + z2[i];
            z2[i] = b2[i] * y - a2[i] * out;

            y = out;
        }

        y_band[b] = static_cast<float>(y);
    }
}


void SosBank::reset()
{
    std::fill(z1.begin(), z1.end(), 0.0);
    std::fill(z2.begin(), z2.end(), 0.0);
}


// ============================================================
// AcousticProcessor
// ============================================================

AcousticProcessor::AcousticProcessor(const std::vector<float>& bA,
                                     const std::vector<float>& aA,
                                     const std::vector<float>& bC,
                                     const std::vector<float>& aC,
                                     const float* sos_data,
                                     int nbands,
                                     int nsec)
{
    filterA_.init_from_ba(bA, aA);
    filterC_.init_from_ba(bC, aC);

    bank_.init_from_flat_sos(sos_data, nbands, nsec);

    y_band_.resize(nbands);
}


int AcousticProcessor::nbands() const
{
    return bank_.nbands;
}


void AcousticProcessor::reset_state()
{
    filterA_.reset();
    filterC_.reset();
    bank_.reset();
}


void AcousticProcessor::process_into(const float* x,
                                     int nsamples,
                                     int window_size,
                                     float C,
                                     bool compute_bands,
                                     float* out,
                                     int ncols)
{
    if (x == nullptr) {
        throw std::runtime_error("Input audio pointer is null");
    }

    if (out == nullptr) {
        throw std::runtime_error("Output pointer is null");
    }

    if (window_size <= 0) {
        throw std::runtime_error("window_size must be positive");
    }

    const int expected_cols = compute_bands ? 5 + bank_.nbands : 5;

    if (ncols != expected_cols) {
        throw std::runtime_error("Invalid output column count");
    }

    reset_state();

    if (nsamples < window_size) {
        return;
    }

    const int nframes = (nsamples - window_size) / window_size + 1;

    const int fast_chunk = window_size / 8;

    if (fast_chunk <= 0) {
        throw std::runtime_error("window_size too small for LAmax/LAmin");
    }

    std::vector<double> sum_bands;

    if (compute_bands) {
        sum_bands.resize(bank_.nbands);
    }

    for (int f = 0; f < nframes; ++f) {
        const int start = f * window_size;
        const int end = start + window_size;

        double sumA = 0.0;
        double sumC = 0.0;
        double sumZ = 0.0;

        std::array<double, 8> fastA{};
        std::array<int, 8> fastCount{};

        if (compute_bands) {
            std::fill(sum_bands.begin(), sum_bands.end(), 0.0);
        }

        for (int n = start; n < end; ++n) {
            const float sample = x[n];

            const float yA = filterA_.step(sample);
            const float yC = filterC_.step(sample);

            sumA += static_cast<double>(yA) * static_cast<double>(yA);
            sumC += static_cast<double>(yC) * static_cast<double>(yC);
            sumZ += static_cast<double>(sample) * static_cast<double>(sample);

            const int local_n = n - start;
            const int fast_idx = local_n / fast_chunk;

            if (fast_idx >= 0 && fast_idx < 8) {
                fastA[fast_idx] +=
                    static_cast<double>(yA) * static_cast<double>(yA);

                fastCount[fast_idx] += 1;
            }

            if (compute_bands) {
                bank_.process_sample(sample, y_band_.data());

                for (int b = 0; b < bank_.nbands; ++b) {
                    const float yb = y_band_[b];

                    sum_bands[b] +=
                        static_cast<double>(yb) * static_cast<double>(yb);
                }
            }
        }

        float* row = out + f * ncols;

        row[0] = level_from_sum_sq(sumA, window_size, C);
        row[1] = level_from_sum_sq(sumC, window_size, C);
        row[2] = level_from_sum_sq(sumZ, window_size, C);

        float LAmax = -std::numeric_limits<float>::infinity();
        float LAmin =  std::numeric_limits<float>::infinity();

        for (int i = 0; i < 8; ++i) {
            if (fastCount[i] <= 0) {
                continue;
            }

            const float level =
                level_from_sum_sq(fastA[i], fastCount[i], C);

            LAmax = std::max(LAmax, level);
            LAmin = std::min(LAmin, level);
        }

        row[3] = LAmax;
        row[4] = LAmin;

        if (compute_bands) {
            for (int b = 0; b < bank_.nbands; ++b) {
                row[5 + b] =
                    level_from_sum_sq(sum_bands[b], window_size, C);
            }
        }
    }
}