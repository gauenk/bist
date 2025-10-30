

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "structs.h"
#include <cfloat>

#include "structs_3d.h"


void manifold_edges(PointCloudData& data);

std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint64_t>,thrust::device_vector<uint32_t>,thrust::device_vector<uint8_t>,thrust::device_vector<uint32_t>>
gather_faces_by_edge(PointCloudData& data);