#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <math.h>

__host__ void update_params(const float* img, const int* spix,
                            spix_params* sp_params,spix_helper* sp_helper,
                            float sigma2_app, const int npixels,
                            const int sp_size, const int nspix_buffer,
                            const int nbatch, const int width, const int nftrs,
                            bool prop_icov);

/* __host__ void update_params_log(const float* img, const int* spix, */
/*                             spix_params* sp_params,spix_helper* sp_helper, */
/*                             float sigma2_app, const int npixels, */
/*                             const int sp_size, const int nspix_buffer, */
/*                             const int nbatch, const int width, const int nftrs, */
/*                             bool prop_icov, Logger* logger); */


__host__ void update_params_summ(const float* img, const int* spix,
                                 spix_params* sp_params,spix_helper* sp_helper,
                                 float sigma_app, const int npixels,
                                 const int nspix_buffer, const int nbatch,
                                 const int width, const int nftrs);

__global__ void clear_fields(spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int nsuperpixel_buffer,
                             const int nftrs);

__global__ void sum_by_label(const float* img, const int* seg,
                             spix_params* sp_params,
                             spix_helper* sp_helper,
                             const int npix, const int nbatch,
                             const int width, const int nftrs);

__host__
void store_sample_sigma_shape(spix_params* sp_params,spix_helper* sp_helper,
                              const int sp_size, const int nspix_buffer);
__global__
void store_sample_sigma_shape_k(spix_params*  sp_params,spix_helper* sp_helper,
                                float sigma_app, const int nsuperpixel_buffer);

__global__ void calc_summ_stats(spix_params*  sp_params,spix_helper* sp_helper,
                                float sigma_app, const int nsuperpixel_buffer);

__global__ void calc_simple_update(spix_params*  sp_params,spix_helper* sp_helper,
                                   float sigma_app, const int sp_size,
                                   const int nsuperpixel_buffer, bool prop_flag);
/* __global__ */
/* void calc_simple_update_log(spix_params*  sp_params,spix_helper* sp_helper, */
/*                             float sigma_app, const int sp_size, */
/*                             const int nsuperpixel_buffer, bool prop_flag, Logger* logger); */

