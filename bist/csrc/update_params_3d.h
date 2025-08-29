
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs_3d.h"

__host__ void update_params(spix_params* aos_params, spix_helper* sm_helper, PointCloudData& data, SuperpixelParams3d& soa_params, SpixMetaData& args, Logger* logger);

__global__
void clear_fields(spix_params* sp_params, spix_helper* sp_helper, const int nspix_total);

__global__
void sum_by_label(const float3* ftrs, const float3* pos,
                  const int* vptr, const uint8_t* vbids,
                  const uint32_t* spix, const uint32_t* csum_nspix,
                  spix_params* sp_params, spix_helper* sp_helper,
                  const int V_total, const int nspix_buffer);

__global__
void update_params_kernel(spix_params* sp_params, spix_helper* sp_helper,
                          float sigma_app, const int sp_size, const int nsuperpixel_buffer);