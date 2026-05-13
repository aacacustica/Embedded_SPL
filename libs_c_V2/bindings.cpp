#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "acoustic_processor.hpp"

namespace py = pybind11;

static std::vector<float> array_1d_to_vector(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{

    auto buf = arr.request();

    if(buf.ndim != 1){
        throw std::runtime_error("Expected 1D float32 array");
    }

    const auto* ptr = static_cast<const float*>(buf.ptr);
    const size_t n = buf.shape[0];

    return std::vector<float>(ptr,ptr + n);

};

PYBIND11_MODULE(leq_levels_oct_weighting_C, m)
{
    py::class_<AcousticProcessor>(m, "AcousticProcessor")
        .def(
            py::init([](
                py::array_t<float, py::array::c_style | py::array::forcecast> bA,
                py::array_t<float, py::array::c_style | py::array::forcecast> aA,
                py::array_t<float, py::array::c_style | py::array::forcecast> bC,
                py::array_t<float, py::array::c_style | py::array::forcecast> aC,
                py::array_t<float, py::array::c_style | py::array::forcecast> sos_bank
            ) {
                auto sos_buf = sos_bank.request();

                if (sos_buf.ndim != 3) {
                    throw std::runtime_error(
                        "sos_bank must have shape (nbands, nsec, 6)"
                    );
                }

                const int nbands = static_cast<int>(sos_buf.shape[0]);
                const int nsec = static_cast<int>(sos_buf.shape[1]);
                const int ncoef = static_cast<int>(sos_buf.shape[2]);

                if (ncoef != 6) {
                    throw std::runtime_error(
                        "sos_bank last dimension must be 6"
                    );
                }

                const auto bA_vec = array_1d_to_vector(bA);
                const auto aA_vec = array_1d_to_vector(aA);
                const auto bC_vec = array_1d_to_vector(bC);
                const auto aC_vec = array_1d_to_vector(aC);

                const auto* sos_ptr =
                    static_cast<const float*>(sos_buf.ptr);

                return std::make_unique<AcousticProcessor>(
                    bA_vec,
                    aA_vec,
                    bC_vec,
                    aC_vec,
                    sos_ptr,
                    nbands,
                    nsec
                );
            }),
            py::arg("bA"),
            py::arg("aA"),
            py::arg("bC"),
            py::arg("aC"),
            py::arg("sos_bank")
        )
        .def(
            "process",
            [](
                AcousticProcessor& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x,
                int window_size,
                float C,
                bool compute_bands
            ) {
                auto x_buf = x.request();

                if (x_buf.ndim != 1) {
                    throw std::runtime_error("x must be 1D float32 array");
                }

                const int nsamples =
                    static_cast<int>(x_buf.shape[0]);

                if (window_size <= 0) {
                    throw std::runtime_error("window_size must be positive");
                }

                int nframes = 0;

                if (nsamples >= window_size) {
                    nframes = (nsamples - window_size) / window_size + 1;
                }

                const int ncols =
                    compute_bands ? 5 + self.nbands() : 5;

                py::array_t<float> out({nframes, ncols});

                auto out_buf = out.request();

                const auto* x_ptr =
                    static_cast<const float*>(x_buf.ptr);

                auto* out_ptr =
                    static_cast<float*>(out_buf.ptr);

                {
                    py::gil_scoped_release release;

                    self.process_into(
                        x_ptr,
                        nsamples,
                        window_size,
                        C,
                        compute_bands,
                        out_ptr,
                        ncols
                    );
                }

                return out;
            },
            py::arg("x"),
            py::arg("window_size"),
            py::arg("C"),
            py::arg("compute_bands")
        );
}