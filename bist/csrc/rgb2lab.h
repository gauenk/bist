
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ void rgb2lab(float* img_rgb, float* img_lab, int nbatch, int npix);
__host__ void lab2rgb(float* img_rgb, float* img_lab, int nbatch, int npix);
__global__ void rgb_to_lab(float* img_rgb,float* img_lab, int npix);
__global__ void lab_to_rgb(float* img_rgb,float* img_lab, int npix);

__host__ float* rescale_image(uint8_t* img_rgb, int nbatch, int npix, float scale);
__global__ void rescale_image_kernel(uint8_t* img_rgb, float* img_rs,int npix, float mval);

