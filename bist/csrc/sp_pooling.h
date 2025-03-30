
// -- cpp imports --
#include <stdio.h>
#include <assert.h>

// -- cuda imports --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

std::tuple<float*,float*,int*>
  run_sp_pooling(float* tensor, int* seg, int nspix,
                 int nbatch, int height, int width, int nftrs);

float* run_sp_upsample(int* seg, float* down,
                       int nspix, int nbatch, int npix, int nftrs);

std::tuple<float*,int*>
run_sp_downsample(float* tensor, int* seg,
                  int nspix, int nbatch, int npix, int nftrs);

__global__
void sp_upsample_kernel(float* pooled, int* seg, float* downsampled,
                        int nspix, int nbatch, int npix, int nftrs);

__global__
void normz_downsample_kernel(float* down, int* counts,
                             int nspix, int nbatch, int nftrs);

__global__
void sp_downsample_kernel(float* tensor, int* seg,
                          float* downsampled, float* downcount,
                          int nspix, int nbatch, int npix, int nftrs);


