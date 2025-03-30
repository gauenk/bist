#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"


std::tuple<int*,bool*,SuperpixelParams*>
  run_prop(float* img, int nbatch, int height, int width, int nftrs,
           int niters, int niters_seg, int sm_start, int sp_size,
           float sigma2_app, float sigma2_size,
           float potts, float alpha_hastings,
           int* spix_prev, int* shifted, SuperpixelParams* params_prev,
           float thresh_relabel, float thresh_new,
           float merge_offset, float split_alpha,
           float iperc_coeff, int target_nspix, bool prop_flag, Logger* logger=nullptr);

/* __host__ */
/* int prop_bass_v2(float* img, int* seg, int* shifted, */
/*                  spix_params* sp_params, */
/*                  bool* border,spix_helper* sp_helper, */
/*                  /\* spix_helper_sm* sm_helper, *\/ */
/*                  spix_helper_sm_v2* sm_helper, */
/*                  int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                  int niters, int niters_seg, int sm_start, */
/*                  float sigma2_app,  float sigma2_size, int sp_size, */
/*                  float potts, float alpha_hastings, */
/*                  int nspix, int nspix_buffer, */
/*                  int nbatch, int width, int height, int nftrs, */
/*                  SuperpixelParams* params_prev, int nspix_prev, */
/*                  float thresh_relabel, float thresh_new, float merge_offset); */
