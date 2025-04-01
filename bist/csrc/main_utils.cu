
/***********************************************************


                   Get Image Border


***********************************************************/

// -- basic --
#include <cmath>
#include <iostream>
#include <fstream>
#include <dirent.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// -- opencv --
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// using namespace cv;
using namespace std;

#include "structs.h"
#include "main_utils.h"
#include "init_utils.h"
#include "sp_pooling.h"
#include "rgb2lab.h"
#include "atomic_helpers.h"
#include "init_seg.h"
#include "compact_spix.h"
#include "seg_utils.h" // dev only

#define THREADS_PER_BLOCK 512


superpixel_options get_sp_options(int nPixels_in_square_side,
                                         float i_std,float beta, float alpha_hasting){
    superpixel_options opt;
    opt.nPixels_in_square_side = nPixels_in_square_side;
    opt.i_std = i_std;
    opt.beta_potts_term = beta;
    opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;
    opt.s_std = opt.nPixels_in_square_side;
    opt.prior_count = opt.area*opt.area ;
    opt.calc_cov = true;
    opt.use_hex = false;
    opt.alpha_hasting = alpha_hasting;

    opt.nEMIters = opt.nPixels_in_square_side;
    // opt.nEMIters = 10000;
    //opt.nEMIters = 15;
    opt.nInnerIters = 4;
    return opt;
}

void show_usage(const std::string &program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -h, --help                Show this help message\n"
              << "  -d, --image_direc DIR     Set image directory (default: image/)\n"
              << "  -o, --output_direc DIR    Output directory (default: result/)\n"
              << "  -n, --nPixels_on_side N   Set number of pixels on side (default: 15)\n"
              << "  --sigma_app VALUE             Set i_std value (default: 0.018)\n"
              << "  --im_size VALUE           Set image size (default: 0)\n"
              << "  --beta VALUE              Set beta value (default: 0.5)\n"
              << "  --alpha VALUE             Set alpha value (default: 0.5)\n"
              << "  --img_ext VALUE           File extension (default: jpg)\n"
              << "  --read_video BOOL         Read a video (true) or images (false)\n"
              << "  --subdir DIR              Set subdirectory (default: none)\n";
}



/********* Create a directory if it doesn't exist. ***********/
void ensureDirectoryExists(const std::string& path) {
  if (access(path.c_str(), F_OK) == -1) { // Check if directory exists
      if (mkdir(path.c_str(), 0755) != 0) {
        throw std::runtime_error("Failed to create directory: " + path);
      }
  }
}



void save_spix_gpu(cv::String fname, int* spix, int height, int width){
  int* spix_cpu = (int*)malloc(height*width*sizeof(int));
  cudaMemcpy(spix_cpu,spix,height*width*sizeof(int),cudaMemcpyDeviceToHost);
  save_spix(fname, spix_cpu, height, width);
  free(spix_cpu);
}

void save_spix(cv::String fname, int* spix, int height, int width){
  std::ofstream file1;
  std::cout << "Writing segmenation to " << fname << std::endl;
  file1.open(fname);
  int idx = 0;
  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      int spix_i = spix[idx];
      // if (spix_i < 0){
      //   printf("ERROR! invalid spix @ (%d,%d)\n",i,j);
      // }
      if (j==width-1){
        file1 << spix_i;
      }else{
        file1 << spix_i <<",";
      }
      idx++;
    }
    file1 << '\n';
  }
  file1.close();

}

void save_params(cv::String fname, SuperpixelParams* sparams){
  // std::vector<int> spix_ids = sparams->ids;
  thrust::host_vector<int> spix_ids = sparams->ids; // convert
  // // Convert to std::vector
  // std::vector<int> std_vec(h_vec.begin(), h_vec.end());

  cout << "Writing parameters to " << fname << endl;
    std::ofstream outfile(fname); // Create an ofstream object to write to a file named "output.txt"
    if (!outfile.is_open()) { // Check if the file opened successfully
      std::cerr << "Error opening output file." << std::endl;
      exit(1);
    }
    int nspix = spix_ids.size();
    // std::set<int> uniqueSet(myVector.begin(), myVector.end());

    outfile << "label,mu_i.x,mu_i.y,mu_i.z,p_mu_i.x,p_mu_i.y,p_mu_i.z,mu_s.x,mu_s.y,sigma_s.x,sigma_s.y,sigma_s.z,prior_icov.x,prior_icov.y,prior_icov.z,count,prior_count" << std::endl; // Write to the file
    // for(int idx=spix_ids.begin(); idx < spix_ids.end(); idx++){

    thrust::host_vector<float> mu_app = sparams->mu_app;
    thrust::host_vector<double> mu_shape = sparams->mu_shape;
    thrust::host_vector<double> sigma_shape = sparams->sigma_shape;
    thrust::host_vector<int> count = sparams->counts;
    thrust::host_vector<float> prior_count = sparams->prior_counts;
    thrust::host_vector<float> prior_mu_app = sparams->prior_mu_app;
    thrust::host_vector<double> prior_sigma_shape = sparams->prior_sigma_shape;


    for(int spix_index : spix_ids){
      if (spix_index < 0){ continue; }
      // int spix_index = spix_ids[idx];
      // superpixel_params sparam = sparams[spix_index];

      float3 mu_i;
      mu_i.x = mu_app[3*spix_index+0];
      mu_i.y = mu_app[3*spix_index+1];
      mu_i.z = mu_app[3*spix_index+2];

      double2 mu_s;
      mu_s.x = mu_shape[2*spix_index+0];
      mu_s.y = mu_shape[2*spix_index+1];

      double3 sigma_s;
      sigma_s.x = sigma_shape[3*spix_index+0];
      sigma_s.y = sigma_shape[3*spix_index+1];
      sigma_s.z = sigma_shape[3*spix_index+2];

      // double3 prior_icov = compute_icov(sigma_s);
      double3 prior_icov;
      prior_icov.x = prior_sigma_shape[3*spix_index+0];
      prior_icov.y = prior_sigma_shape[3*spix_index+1];
      prior_icov.z = prior_sigma_shape[3*spix_index+2];

      float3 p_mu_i;
      p_mu_i.x = prior_mu_app[3*spix_index+0];
      p_mu_i.y = prior_mu_app[3*spix_index+1];
      p_mu_i.z = prior_mu_app[3*spix_index+2];
      
      int _count = count[spix_index];
      float _prior_count = prior_count[spix_index];

      // Write the data in CSV format
      outfile << spix_index << "," << mu_i.x << "," << mu_i.y << "," << mu_i.z << ","
              << p_mu_i.x << "," << p_mu_i.y << "," << p_mu_i.z << ","
              << mu_s.x << "," << mu_s.y << ","
              << sigma_s.x << "," << sigma_s.y << "," << sigma_s.z << ","
              << prior_icov.x << "," << prior_icov.y << "," << prior_icov.z << ","
              << _count << "," << _prior_count << std::endl;
    }
    outfile.close(); // Close the file
}

double3 compute_icov(double3 cov){
  // -- correct sample cov if not invertable --
  if (abs(cov.x) < 1e-8){
    cov.x = 1;
  }else if (abs(cov.z) < 1e-8){
    cov.z = 1;
  }

  double3 icov;
  double det = cov.x*cov.z - cov.y*cov.y;
  if (det <= 0){
    cov.x = cov.x + 0.00001;
    cov.z = cov.z + 0.00001;
    det = cov.x * cov.z - cov.y * cov.y;
    if (det<=0){ det = 0.00001; } // safety hack
  }
  icov.x = cov.z/det;
  icov.y = -cov.y/det;
  icov.z = cov.x/det;
  return icov;
}

cv::Mat get_img_res(float* img0, float* img1, int height, int width){

  // -- info --
  int nftrs = 3; 
  int nbatch = 1;
  int npix = height * width;

  // -- create delta --
  float* delta = (float*)easy_allocate(nftrs * npix,sizeof(float));
  float* dmax = (float*)easy_allocate(1,sizeof(float));
  cudaMemset(dmax,0,sizeof(float));
  compute_delta(delta, img0, img1, npix, dmax);

  // -- normalize and re-format --
  uint8_t* delta_fmt = (uint8_t*)easy_allocate(nftrs * npix,sizeof(uint8_t));
  normz_and_format(delta,delta_fmt,nftrs*npix,dmax);

  // -- copy to cv2 image --
  uint8_t* delta_cpu = (uint8_t*)malloc(nftrs * npix * sizeof(uint8_t));
  cudaMemcpy(delta_cpu, delta_fmt,
             nftrs * npix * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cv::Mat img_delta(height, width, CV_8UC3, delta_cpu);

  // -- free --
  cudaFree(delta_fmt);
  cudaFree(dmax);
  cudaFree(delta);

  return img_delta;

}

__host__ void normz_and_format(float* delta, uint8_t* delta_fmt, int npix, float* dmax){
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  normz_and_format_k<<<BlockPerGrid,ThreadPerBlock>>>(delta, delta_fmt, npix, dmax);
}

__global__
void normz_and_format_k(float* delta, uint8_t* delta_fmt, int npix, float* dmax){
  // -- indexing --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (ix>=npix) return;
  delta_fmt[ix] = __float2uint_rn(255*delta[ix]/(*dmax));

}

__host__ void compute_delta(float* delta, float* img0, float* img1,
                                 int npix, float* dmax){
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  compute_delta_k<<<BlockPerGrid,ThreadPerBlock>>>(delta, img0, img1, npix, dmax);
}

__global__ void compute_delta_k(float* delta, float* img0, float* img1,
                              int npix, float* dmax){

  // -- indexing --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (ix>=npix) return;

  // -- compute difference --
  float _delta = 0;
  float _dj = 0;
  for(int jx=0;jx<3;jx++){
    _dj = img0[ix*3+jx] - img1[ix*3+jx];
    _delta += _dj*_dj;
  }
  delta[3*ix+0] = _delta;
  delta[3*ix+1] = _delta;
  delta[3*ix+2] = _delta;
  atomicMaxFloat(dmax,_delta);
}

std::tuple<cv::Mat,float*>
get_img_pooled(float* img, int* spix, int height, int width, int nspix){

  // -- info --
  int nftrs = 3; 
  int nbatch = 1;
  int npix = height * width;

  // -- run pooling --
  auto out = run_sp_pooling(img,spix,nspix,nbatch,height,width,nftrs);
  float* pooled = std::get<0>(out);
  float* down = std::get<1>(out);
  int* counts = std::get<2>(out);

  // -- normalize and re-format --
  float* dmax = (float*)easy_allocate(1,sizeof(float));
  float one = 1.0f;
  cudaMemcpy(dmax, &one, sizeof(float), cudaMemcpyHostToDevice);
  uint8_t* pooled_fmt = (uint8_t*)easy_allocate(nftrs * npix,sizeof(uint8_t));
  normz_and_format(pooled,pooled_fmt,nftrs*npix,dmax);

  // -- copy to cv2 image --
  uint8_t* pooled_cpu = (uint8_t*)malloc(nftrs * npix * sizeof(uint8_t));
  cudaMemcpy(pooled_cpu, pooled_fmt,
             nftrs * npix * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cv::Mat img_pooled(height, width, CV_8UC3, pooled_cpu);

  // -- free --
  cudaFree(pooled_fmt);
  cudaFree(counts);
  cudaFree(down);
  // cudaFree(pooled);

  return std::make_tuple(img_pooled,pooled);
}


cv::Mat get_img_border(float* img, bool* border, int height, int width, int nftrs){
  int npix = height * width;
  uint8_t* image_border_cpu = (uint8_t*)malloc(nftrs * npix * sizeof(uint8_t));
  uint8_t* filled_img = (uint8_t*)easy_allocate(nftrs * npix,sizeof(uint8_t));

  CUDA_get_image_overlaid(filled_img, img, border, npix, width);
  cudaMemcpy(image_border_cpu, filled_img,
             nftrs * npix * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cv::Mat img_border(height, width, CV_8UC3, image_border_cpu);

  // free(image_border_cpu);
  cudaFree(filled_img);
  return img_border;
}

__host__ void
CUDA_get_image_overlaid(uint8_t* filled, float* image,
                        const bool* border, const int nPixels, const int xdim){
  int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  GetImageOverlaid<<<BlockPerGrid,ThreadPerBlock>>>(filled, image, border, nPixels, xdim);
}

__global__ void GetImageOverlaid(uint8_t* filled, float* image, const bool* border,
                                 const int nPixels, const int xdim){
  int t = threadIdx.x + blockIdx.x * blockDim.x;  
  if (t>=nPixels) return;

  if (border[t]){
    // -- for a nice grey --
    // filled[3*t] = 50;
    // filled[3*t+1] = 50;
    // filled[3*t+2] = 50;
    // -- for a sharp blue --
    filled[3*t] = 255; // blue
    filled[3*t+1] = 0;
    filled[3*t+2] = 0;
  }else{
    filled[3*t] = min(255*image[3*t],255.);
    filled[3*t+1] = min(255*image[3*t+1],255.);
    filled[3*t+2] = min(255*image[3*t+2],255.);
  }
    
}


std::tuple<int*,int,bool*>
get_square_segmentation(int sp_size, int nbatch, int height, int width){

  // -- init spix --
  int npix = height*width;
  int* _spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
  cudaMemset(_spix,-1,sizeof(int));
  int nspix = init_square_seg(_spix, sp_size, width, height, nbatch);

 //  // -- get max nspix --
 //  thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
 //  thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);
 //  int max_spix = *thrust::max_element(thrust::device, spix.begin(), spix.end());
 //  int min_spix = *thrust::min_element(thrust::device, spix.begin(), spix.end());

 //  // -- dummy params for lazy coding --
 //  int nspix_buffer = nspix*30;
 //  const int sparam_size = sizeof(spix_params);
 //  spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);

 // // -- compactify --
 //  thrust::device_vector<int> prop_ids = extract_unique_ids(_spix, npix, 0);
 //  nspix = compactify_new_superpixels(_spix,sp_params,prop_ids,0,max_spix,npix);
 //  cudaFree(sp_params);

  // -- get border --
  bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
  CudaFindBorderPixels(_spix,border,npix,nbatch,width,height);

  return std::make_tuple(_spix,nspix,border);

}
