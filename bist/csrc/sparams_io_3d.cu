/************************************************************

     "Read" means we go from spix_params* to SuperpixelParams
     "Write" means we go from SuperpixelParams to spix_params* 

*************************************************************/


// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>


#define THREADS_PER_BLOCK 512

#include "sparams_io_3d.h"
#include "structs_3d.h"


/***********************************************************************

                       Spix Parameter Management

***********************************************************************/

__global__
void aos_to_soa_kernel(spix_params* aos_params, float3* mu_app, double3* mu_pos, 
                       double3* var_pos, double3* cov_pos, int nspix) {


    // -- filling superpixel params into image --
    int spix_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (spix_index >= nspix) return;
    spix_params param = aos_params[spix_index];

    // -- copy over --
    mu_app[spix_index] = param.mu_app;
    mu_pos[spix_index] = param.mu_pos;
    var_pos[spix_index] = param.var_pos;
    cov_pos[spix_index] = param.cov_pos;

}

// Array of Structures -> Structures of Arrays [spix_params -> SuperpixelParams3d]
__host__ void aos_to_soa(spix_params* aos_params, SuperpixelParams3d& soa_params){
  
  // -- unpack pointers --
  int nspix_max = soa_params.nspix_sum;
  soa_params.resize_spix_params(nspix_max);
  float3* mu_app = thrust::raw_pointer_cast(soa_params.mu_app.data());
  double3* mu_pos = thrust::raw_pointer_cast(soa_params.mu_pos.data());
  double3* var_pos = thrust::raw_pointer_cast(soa_params.var_pos.data());
  double3* cov_pos = thrust::raw_pointer_cast(soa_params.cov_pos.data());

  // -- write from [mu_app,mu_shape,...] to [sp_params] --
  int num_blocks = ceil( double(nspix_max) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  aos_to_soa_kernel<<<nblocks,nthreads>>>(aos_params, mu_app, mu_pos, var_pos, cov_pos,  nspix_max);

}
