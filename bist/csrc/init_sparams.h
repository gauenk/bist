/*************************************************

        This script initializes spix_params
            WITH and WITHOUT propogation

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"


/*************************************************

                WITHOUT prior

**************************************************/

__global__ void mark_inactive_kernel(spix_params* params, int nspix_buffer, int*nvalid,
                                     int nspix, int sp_size);
__global__ void mark_active_kernel(spix_params* params, int* ids, int nactive,
                                   int nspix, int spix_buffer, int*nvalid);
__host__ void mark_active(spix_params* params, int* ids, int nactive,
                          int nspix, int nspix_buffer, int sp_size);
__host__ void mark_active_contiguous(spix_params* params, int nspix,
                                     int nspix_buffer, int sp_size);


__host__ void init_sp_params(spix_params* sp_params, float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                             int npix, int nspix, int nspix_buffer,
                             int nbatch, int width, int nftrs, int sp_size);

__host__ void init_sp_params_b(spix_params* sp_params, float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                              thrust::device_vector<int>& nspix, int nspix_buffer, int npix,
                             int nbatch, int width, int nftrs, int sp_size);


__global__ void init_sp_params_kernel(spix_params* sp_params,float prior_sigma_app,
                                      const int nspix, int nspix_buffer,
                                      int npix, int sp_size);
__global__ void init_sp_params_kernel_b(spix_params* sp_params,float prior_sigma_app,
                                       const int* nspix, int nspix_buffer,
                                       int npix, int sp_size);

__host__
void init_sp_params_from_past(spix_params* curr_params,
                              spix_params* prev_params,
                              int* curr2prev_map, float4 rescale,
                              int nspix, int nspix_buffer,int npix);

__global__
void init_sp_params_from_past_kernel(spix_params* curr_params,
                                     spix_params* prev_params,
                                     /* int* map, */
                                     float4 rescale, int nspix,
                                     int nspix_buffer, int npix);
