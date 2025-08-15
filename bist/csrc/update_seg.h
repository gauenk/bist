
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "seg_utils.h"

__host__ void set_border(int* seg, bool* border, int height, int width);

__global__
void update_seg_subset(float* img, int* seg, bool* border,
                       spix_params* sp_params, const float sigma2_app,
                       const float potts, const int npix, const int nspix_buffer,
                       const int width, const int height, const int nftrs,
                       const int xmod3, const int ymod3);

__host__
void update_seg(float* img, int* seg, bool* border,
                spix_params* sp_params, const int niters,
                const float sigma2_app, const float potts,
                const int npix, int nspix_buffer,
		int nbatch, int width, int height, int nftrs,
		Logger* logger=nullptr);

// __device__ float2 calc_joint(float* imgC, int* seg,
//                              int width_index, int height_index,
//                              spix_params* sp_params,
// 			     int seg_label, int seg_index,
// 			     float sigma2_app,  float neigh_neq,
//                              float beta, float2 res_max);
__device__ float2 calc_joint(float* imgC, int* seg,
                             int width_index, int height_index,
                             spix_params* sp_params,
			      int seg_index, int seg_label,
			     float sigma2_app,  float neigh_neq,
                             float beta, float2 res_max);

/* __device__ float2 calc_joint(float* imgC, int* seg, */
/*                              int width_index, int height_index, */
/*                              spix_params* sp_params, int seg_idx, */
/*                              float sigma2_app,  float neigh_neq, */
/*                              float beta, float2 res_max); */


/* __host__ */
/* void update_seg_log(float* img, int* seg, bool* border, */
/*                     spix_params* sp_params, const int niters, */
/*                     const float sigma2_app, const float potts, */
/*                     const int npix, int nbatch, */
/*                     int width, int height, int nftrs, Logger* logger); */

/* __global__ */
/* void update_seg_subset_log(float* img, int* seg, bool* border, */
/*                            spix_params* sp_params, */
/*                            const float sigma2_app, const float potts, */
/*                            const int npix, const int nbatch, */
/*                            const int width, const int height, const int nftrs, */
/*                            const int xmod3, const int ymod3, Logger* logger); */

