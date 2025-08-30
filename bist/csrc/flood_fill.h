
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>


#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "cuda.h"
#include "cuda_runtime.h"
// #include "atomic_helpers.h"
#include "structs_3d.h"

void flood_fill(PointCloudData& data, spix_params* params, uint32_t* spix, uint32_t* csum_nspix, uint32_t S);
 

__global__ void
find_clostest_point(uint64_t* distances, float3* pos, spix_params* params, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V);


__global__ void
set_only_seeds(uint64_t* distances, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V);


__global__ void
propogate_seed(uint32_t* spix, bool* valid_label, uint32_t* edges, uint32_t* eptr, uint32_t V);

