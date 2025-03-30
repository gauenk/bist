/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
namespace py = pybind11;

// -- fxns --
void init_bist(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_bist(m);
}
