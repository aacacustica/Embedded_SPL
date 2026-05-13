#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

// ============================================================
// IIRFilterDF2T
// ============================================================
//
// Filtro IIR general equivalente a scipy.signal.lfilter
// usando Direct Form II Transposed.
//
// Soporta filtros de orden arbitrario:
//
//     b = [b0, b1, ..., bM]
//     a = [a0, a1, ..., aN]
//
// En tu caso:
//     A_weighting -> len(b)=7, len(a)=7
//     C_weighting -> len(b)=5, len(a)=5
//
struct IIRFilterDF2T {
    std::vector<double> b;
    std::vector<double> a;
    std::vector<double> z;

    void init_from_ba(const std::vector<float>& b_in,
                      const std::vector<float>& a_in);

    float step(float x);

    void reset();
};


// ============================================================
// Biquad
// ============================================================
//
// Representa una sección SOS:
//
//     b0 + b1 z^-1 + b2 z^-2
// H = -------------------------
//     a0 + a1 z^-1 + a2 z^-2
//
// Internamente normalizamos por a0, por eso guardamos:
// b0, b1, b2, a1, a2
//
// Implementación: Direct Form II Transposed.
// Es eficiente porque solo necesita dos estados: z1, z2.
//
struct Biquad {
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;

    double z1 = 0.0;
    double z2 = 0.0;

    void init(double raw_b0,
              double raw_b1,
              double raw_b2,
              double raw_a0,
              double raw_a1,
              double raw_a2);

    float step(float x);

    void reset();
};


// ============================================================
// BiquadCascade
// ============================================================
//
// Cadena de biquads.
// Sirve para filtros A/C y, si quisieras, cualquier filtro SOS.
//
struct BiquadCascade {
    std::vector<Biquad> sections;

    // Para filtros definidos como b/a de tamaño 3:
    // b = [b0, b1, b2]
    // a = [a0, a1, a2]
    void init_from_ba(const std::vector<float>& b,
                      const std::vector<float>& a);

    // Para filtros SOS en formato plano:
    //
    // sos_flat = [
    //   b0, b1, b2, a0, a1, a2,
    //   b0, b1, b2, a0, a1, a2,
    //   ...
    // ]
    void init_from_sos_flat(const std::vector<float>& sos_flat,
                            int n_sections);

    float step(float x);

    void reset();
};


// ============================================================
// SosBank
// ============================================================
//
// Banco de filtros de bandas de tercio de octava.
//
// Entrada esperada desde Python:
//     sos_bank.shape == (nbands, nsec, 6)
//
// Por ejemplo, en tu caso:
//     nbands = 25
//     nsec   = 4
//     6 coeficientes por sección:
//         b0, b1, b2, a0, a1, a2
//
// Internamente lo almacenamos como arrays separados:
//
//     b0[], b1[], b2[], a1[], a2[], z1[], z2[]
//
// Esto suele ser más cómodo y eficiente para recorrer en bucles.
//
struct SosBank {
    int nbands = 0;
    int nsec = 0;

    std::vector<double> b0;
    std::vector<double> b1;
    std::vector<double> b2;
    std::vector<double> a1;
    std::vector<double> a2;

    std::vector<double> z1;
    std::vector<double> z2;

    int idx(int band, int section) const;

    void init_from_flat_sos(const float* data,
                            int bands,
                            int sections);

    void process_sample(float x, float* y_band);

    void reset();
};


// ============================================================
// AcousticProcessor
// ============================================================
//
// Clase principal expuesta a Python mediante pybind11.
//
// Uso previsto:
//
//   processor = AcousticProcessor(bA, aA, bC, aC, sos_bank)
//   levels = processor.process(x, window_size, C, compute_bands)
//
// Pero la función Python realmente llamará a process_into()
// desde bindings.cpp.
//
// Importante:
//   - Los coeficientes se cargan una vez.
//   - Los estados se resetean al inicio de cada archivo.
//   - El bucle pesado queda dentro de C++.
//   - En modo no_bands no se procesan las 25 bandas.
//
class AcousticProcessor {
public:
    AcousticProcessor(const std::vector<float>& bA,
                      const std::vector<float>& aA,
                      const std::vector<float>& bC,
                      const std::vector<float>& aC,
                      const float* sos_data,
                      int nbands,
                      int nsec);

    int nbands() const;

    void reset_state();

    void process_into(const float* x,
                      int nsamples,
                      int window_size,
                      float C,
                      bool compute_bands,
                      float* out,
                      int ncols);
private:
    IIRFilterDF2T filterA_;
    IIRFilterDF2T filterC_;
    SosBank bank_;

    std::vector<float> y_band_;
};