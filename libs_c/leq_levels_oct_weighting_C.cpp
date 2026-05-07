#include <iostream>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.cpp"

using namespace std;

float PREF = 2e-6;
float EPS = 1e-30; //EPSILON TO AVOID BY-ZERO OPERATIONS

int main(){
    cout << "Audio processing module loaded susccesfully" << endl;
    return 0;
}

PYBIND11_MODULE(leq_levels_oct_weighting_C, m) {
    m.doc() = "C++17 module for Leq level octave weighting calculations";
    m.def("get_level_db", &get_level_db, "Calculate level in dB from signal array and correction factor C",
          pybind11::arg("x"), pybind11::arg("C"));
    m.def("lfilter_np", &lfilter_np, "Apply a digital filter to a signal using given coefficients and initial state",
          pybind11::arg("b"), pybind11::arg("a"), pybind11::arg("x"), pybind11::arg("zi") = pybind11::array_t<float>());
    m.def("sosfilt_np", &sosfilt_np, "Apply a digital filter in SOS format to a signal using given initial state",
          pybind11::arg("sos"), pybind11::arg("x"), pybind11::arg("zi") = pybind11::array_t<float>());
}   