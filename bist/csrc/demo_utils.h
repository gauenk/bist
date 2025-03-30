
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


// using namespace cv;
using namespace std;


// // -- basic --
// template <typename T>
// bool parse_argument(int &i, int argc, char **argv, const std::string &arg,
//                     const std::string &option, T &value) {
//     if (arg == option) {
//         if (i + 1 < argc) {
//             ++i;
//             if constexpr (std::is_same<T, int>::value) {
//                 value = std::stoi(argv[i]);
//             } else if constexpr (std::is_same<T, float>::value) {
//                 value = std::stof(argv[i]);
//             } else if constexpr (std::is_same<T, const char *>::value || std::is_same<T, std::string>::value) {
//                 value = argv[i];
//             }
//         } else {
//             std::cerr << option << " option requires an argument." << std::endl;
//             return false;
//         }
//     }
//     return true;
// }


// -- standard functions --
void show_usage(const std::string &program_name);
superpixel_options get_sp_options(int sp_size,float i_std,
                                         float beta, float alpha_hasting);
double3 compute_icov(double3 cov);




// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void save_spix_gpu(cv::String fname, int* spix, int height, int width);
void save_spix(cv::String fname, int* spix, int height, int width);
void save_params(cv::String fname, SuperpixelParams* sparams);

cv::Mat get_img_border(float* img_gpu, bool* border, int height, int width, int nftrs);
__host__ void CUDA_get_image_overlaid(uint8_t* filled, float* image, const bool* border,
                                      const int nPixels, int xdim);
__global__ void GetImageOverlaid(uint8_t* filled, float* image, const bool* border,
                                 const int nPixels, int xdim);

std::tuple<cv::Mat,float*>
get_img_pooled(float* img, int* spix, int height, int width, int nspix);

cv::Mat get_img_res(float* img0, float* img1, int height, int width);
__host__ void compute_delta(float* delta, float* img0, float* img1,int npix, float* dmax);
__global__
void compute_delta_k(float* delta, float* img0, float* img1, int npix, float* dmax);
__host__ void normz_and_format(float* delta, uint8_t* delta_fmt, int npix, float* dmax);
__global__
void normz_and_format_k(float* delta, uint8_t* delta_fmt, int npix, float* dmax);

std::tuple<int*,int,bool*>
get_square_segmentation(int sp_size, int nbatch, int height, int width);







