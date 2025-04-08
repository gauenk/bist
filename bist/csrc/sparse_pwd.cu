/********************************************************************

      Pairwise differences between pixels and superpixels

********************************************************************/

// -- pytorch api --
#include <torch/extension.h>
#include <torch/types.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512


// -- cpp imports --
#include <stdio.h>
#include <assert.h>
#include <tuple>

// -- thrust --
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h> // for debugging

#define THREADS_PER_BLOCK 512


__global__
void sparse_pwd(float* delta, float* video, float* down, int* spix,
                int npix, int nftrs, int nspix, int ksize){

  // -- pixel index --
  int pix_ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (pix_ix>=npix) return;
  int batch_ix = blockIdx.y;
  // pix_ix = pix_ix + npix*batch_ix; // offset via batch

  // -- kernel index --
  int kernel_ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (kernel_ix>=ksize) return;

  // -- read spix id --
  int spix_ix = spix[kernel_ix+pix_ix*ksize+batch_ix*npix*ksize];
  if (spix_ix < 0){ return; }

  // -- compute delta --
  float __delta = 0;
  float _delta = 0;
  for(int fidx = 0; fidx < nftrs; fidx++){
    int down_ix = fidx+spix_ix*nftrs+batch_ix*nspix*nftrs;
    int video_ix = fidx+pix_ix*nftrs+batch_ix*npix*nftrs;
    __delta = video[video_ix]-down[down_ix];
    _delta += __delta*__delta;
  }

  // -- read spix location --
  delta[kernel_ix+pix_ix*ksize+batch_ix*npix*ksize] = _delta;

}

torch::Tensor
sparse_pwd_py(torch::Tensor video, torch::Tensor down,
                  torch::Tensor spix_read){

  // -- check --
  CHECK_INPUT(video);
  CHECK_INPUT(down);
  CHECK_INPUT(spix_read);

  // -- unpack shape --
  int nbatch = video.size(0);
  int height = video.size(1);
  int width = video.size(2);
  int nftrs = video.size(3);
  int npix = height*width;
  int ksize = spix_read.size(3); // T,H,W,K
  int nspix = down.size(1); // T,S,F
  assert(down.size(2) == video.size(3));

  // -- allocate differences --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32)
    .layout(torch::kStrided).device(video.device());
  torch::Tensor delta = -torch::ones({nbatch, height, width, ksize},options_f32);

  // -- unpack --
  float* delta_ptr = delta.data_ptr<float>();
  float* video_ptr = video.data_ptr<float>();
  float* down_ptr = down.data_ptr<float>();
  int* spix_read_ptr = spix_read.data_ptr<int>();

  // -- run pairwise comparison --
  int npix_threads = ceil( double(THREADS_PER_BLOCK) / double(ksize) ); 
  int num_block = ceil( double(npix) / double(npix_threads) ); 
  dim3 ThreadPerBlock(npix_threads,ksize);
  dim3 BlockPerGrid(num_block,nbatch);
  sparse_pwd<<<BlockPerGrid,ThreadPerBlock>>>(delta_ptr,video_ptr,
                                              down_ptr,spix_read_ptr,
                                              npix,nftrs,nspix,ksize);

  return delta;
}

