#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"


std::tuple<int*,bool*,SuperpixelParams*>
  run_bist(float* img, int nbatch, int height, int width, int nftrs,
           int niters, int niters_seg, int sm_start, int sp_size,
           float sigma2_app, float sigma2_size,
           float potts, float alpha_hastings,
           int* spix_prev, int* shifted, SuperpixelParams* params_prev,
           float epsilon_reid, float epsilon_new,
           float merge_offset, float split_alpha,
           float gamma, int target_nspix, bool prop_flag, Logger* logger=nullptr);
