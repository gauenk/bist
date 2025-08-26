

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cfloat>

#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

thrust::device_vector<uint32_t> extract_edges_from_pairs(std::vector<uint32_t>& e0, std::vector<uint32_t>& e1);
