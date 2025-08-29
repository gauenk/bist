#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "seg_utils_3d.h"


__host__ void update_seg(spix_params* aos_params, spix_helper* sp_helper, PointCloudData& data, SuperpixelParams3d& soa_params, SpixMetaData& args, Logger* logger);


__global__
void update_seg_subset(spix_params* params, uint32_t* spix, 
                        const bool* border, const bool* is_simple,
                        const float3* ftrs, const float3* pos,
                        const uint8_t* neigh_neq, const uint8_t* gcolors, const int gcolor, 
                        const uint32_t* csr_edges, const uint32_t* csr_ptr, 
                        const uint32_t V, const float sigma2_app, const float potts_coeff);


__device__ float compute_lprob(float3& ftrs, float3& pos, float neigh_neq,
                               spix_params* sp_params, uint32_t sp_index,
			                   float sigma2_app, float beta);