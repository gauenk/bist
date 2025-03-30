#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"

// -- thrust --
#include <thrust/device_vector.h>


#ifndef SEG_UTILS_H
#define SEG_UTILS_H

// bass table; decodes if a point is a "simple" based on neighbors
__device__ const bool bass_lut[256] = {
           0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
	       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
	       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
	       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
	       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
	       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
	       0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0};


__device__ inline int ischangbale_by_nbrs(bool* nbrs){
  int num = 0;
  int count_diff = 0;
#pragma unroll
   for (int i=7; i>=0; i--)
   {
      num <<= 1;
      if (nbrs[i]) num++;
      else count_diff++;
   }
   // convert binary string "nbrs" into decimal "num"
   /* printf("bass_lut num: %d\n",num); */
   nbrs[8] = bass_lut[num];
   return count_diff;
}



/*
* Set the elements in nbrs "array" to 1 if corresponding neighbor pixel has the same superpixel as "label"
*/
__device__ inline
void set_nbrs(int NW,int N,int NE,int W,
              int E,int SW,int S,int SE,
              int label, bool* nbrs){
  nbrs[0] = (label ==NW);
  nbrs[1] = (label == N);
  nbrs[2] = (label == NE);
  nbrs[3] = (label == W);
  nbrs[4] = (label == E);
  nbrs[5] = (label == SW);
  nbrs[6] = (label == S);
  nbrs[7] = (label == SE);
  return;
}

__device__ inline
void set_nbrs_v1(int NW,int N,int NE,int W,
              int E,int SW,int S,int SE,
              int label, bool* nbrs){
  nbrs[0] = (label ==NW)  || (NW==-1);
  nbrs[1] = (label == N)  || (N ==-1);
  nbrs[2] = (label == NE) || (NE==-1);
  nbrs[3] = (label == W)  || (W ==-1);
  nbrs[4] = (label == E)  || (E ==-1);
  nbrs[5] = (label == SW) || (SW==-1);
  nbrs[6] = (label == S)  || (S ==-1);
  nbrs[7] = (label == SE) || (SE==-1);
  return;
}

/* __device__ inline float2 cal_prop_likelihood( */
/*     float* imgC, int* seg, int width_index, int height_index, */
/*     superpixel_params* sp_params, int seg_idx, */
/*     float3 pix_var, float logdet_pix_var, */
/*     float neigh_neq, float beta, float2 res_max){ */

/*     // -- init res -- */
/*     float res = -1000; // some large negative number // why? */

/*     // -- compute color/spatial differences -- */
/*     const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_i.x); */
/*     const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_i.y); */
/*     const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_i.z); */
/*     const int d0 = width_index - __ldg(&sp_params[seg_idx].mu_s.x); */
/*     const int d1 = height_index - __ldg(&sp_params[seg_idx].mu_s.y); */

/*     // -- color component -- */
/*     const float pix_var_x = pix_var.x; */
/*     const float pix_var_y = pix_var.y; */
/*     const float pix_var_z = pix_var.z; */
/*     const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_s.x); */
/*     const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_s.y); */
/*     const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_s.z); */
/*     const float logdet_sigma_s = __ldg(&sp_params[seg_idx].logdet_Sigma_s); */

/*     // -- color component -- */
/*     res = res - x0*x0*pix_var_x - x1*x1*pix_var_y - x2*x2*pix_var_z; */
/*     res = res - logdet_pix_var; // okay; log p(x,y) = -log detSigma */

/*     // -- space component -- */
/*     res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y; // sign(s_y) = -1 */
/*     res = res - logdet_sigma_s; */

/*     // -- potts term -- */
/*     res = res - beta*neigh_neq; */

/*     // -- update res -- */
/*     if( res>res_max.x ){ */
/*       res_max.x = res; */
/*       res_max.y = seg_idx; */
/*     } */

/*     return res_max; */
/* } */

#endif

thrust::device_vector<int> get_unique(int* spix, int size);

__host__ void set_border(int* seg, bool* border, int height, int width);

__global__
void get_spix_sizes_kernel(int* spix, int* sizes, int npix, int nspix);
int* get_spix_sizes(int* spix, int nbatch, int npix, int nspix);
thrust::device_vector<int> get_spix_counts(int* spix, int nbatch, int npix, int nspix);

__global__
void read_prior_counts_kernel(float* prior_counts,spix_params* params, int nspix_buffer);
thrust::device_vector<float> get_prior_counts(spix_params* params, int nspix_buffer);

__host__ void CudaFindBorderPixels( const int* seg, bool* border, const int nPixels,
                                    const int nbatch, const int xdim, const int ydim);
__host__ void CudaFindBorderPixels_end( const int* seg, bool* border, const int nPixels,
                                        const int nbatch, const int xdim, const int ydim);
__global__  void find_border_pixels( const int* seg, bool* border,
                                     const int nPixels, const int xdim, const int ydim);
__global__  void find_border_pixels_end(const int* seg, bool* border,
                                        const int nPixels,const int xdim,const int ydim);


void view_invalid(spix_params* sp_params, int nspix);
__global__ void view_invalid_kernel(spix_params* sp_params, int nspix);
int count_invalid(int* _spix,int size);

