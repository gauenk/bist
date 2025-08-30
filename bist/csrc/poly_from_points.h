
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>


#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "structs_3d.h"


std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint32_t>>
poly_from_points(PointCloudData& data, SuperpixelParams3d& params, bool* border);
 
__global__ void
list_vertices_by_spix(uint32_t* list_of_vertices, uint32_t* spix, bool* border, uint8_t* bids, 
                      uint32_t* csum_nspix, uint32_t* access_offsets, uint32_t* nvertex_by_spix_csum, uint32_t V);
__global__ void
nvertices_per_spix_kernel(uint32_t* nvertex_by_spix, uint32_t* spix, bool* border, uint32_t* csum_nspix, uint8_t* bids, uint32_t V);

__global__ void
reorder_vertices_backtracking_cycle(uint32_t* list_of_vertices, uint32_t* csr_edges, uint32_t* csr_eptr,
                                   uint8_t* bids, uint32_t* csum_nspix, uint32_t* nvertex_by_spix_csum,
                                   uint32_t* global_ptr, float3* pos, uint32_t V, uint32_t S);