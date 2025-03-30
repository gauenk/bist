

// -- cpp imports --
#include <stdio.h>
#include <assert.h>

// -- cuda imports --
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#define THREADS_PER_BLOCK 512

// -- project imports --
#include "structs.h"
#include "init_utils.h"
#include "sp_pooling.h"

/********************************************


                Forward


********************************************/

__global__
void sp_downsample_kernel(float* tensor, int* seg,
                       float* downsampled, int* downcount,
                       int nspix, int nbatch, int npix, int nftrs){
  
  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  pix_idx = pix_idx + npix*blockIdx.y;

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  seg_idx = seg_idx + nspix*blockIdx.y;
  if (seg_idx < 0){ return; }

  // -- add to downsampled --
  float* tensorF = tensor + pix_idx * nftrs;
  float* dsF = downsampled + (seg_idx + nspix*blockIdx.y)*nftrs;
  int* dsC = downcount + seg_idx + nspix*blockIdx.y;
  for (int fidx = 0; fidx < nftrs; fidx++){
    float val = tensorF[fidx];
    // if (isnan(val)){
    //   printf("NAN! pix_idx: %d,%d\n",pix_idx/320,pix_idx%320);
    // }
    atomicAdd(dsF+fidx,*(tensorF+fidx));
  }
  atomicAdd(dsC,1);
}

__global__
void normz_downsample_kernel(float* down, int* counts,
                             int nspix, int nbatch, int nftrs){

  // -- get superpixel index --
  int spix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_idx>=nspix) return;
  spix_idx = spix_idx + nspix*blockIdx.y;

  // -- normalize --
  float* dsF = down + (spix_idx)*nftrs;
  int count = counts[spix_idx];
  for (int fidx = 0; fidx < nftrs; fidx++){
    dsF[fidx] = (count > 0) ? dsF[fidx]/(1.*count) : 0;
  }

}


__global__
void sp_upsample_kernel(float* pooled, int* seg, float* downsampled,
                        int nspix, int nbatch, int npix, int nftrs){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  pix_idx = pix_idx + npix*blockIdx.y;

  // -- get segmentation index --
  int seg_idx = seg[pix_idx];
  if (seg_idx < 0){ return; }
  seg_idx = seg_idx + nspix*blockIdx.y;
  // if (seg_idx >= nspix){
  //   printf("(nspix,seg_idx): (%d,%d)\n",nspix,seg_idx);
  // }
  assert(seg_idx < nspix);

  // -- write to pooled --
  float* poolF = pooled + pix_idx * nftrs;
  float* dsF = downsampled + seg_idx * nftrs;
  for (int fidx = 0; fidx < nftrs; fidx++){
    poolF[fidx] = dsF[fidx];
    // *(poolF+fidx) = *(dsF+fidx);
  }

  // // -- info [remove me!] --
  // int width = 854;
  // int w_idx = pix_idx % width;
  // int h_idx = pix_idx / width;
  // if ((h_idx == 211) and (w_idx == 532)){
  // // if ((h_idx == 0) and (w_idx == 0)){
  //   // printf("flow[211,532]: %2.3f,%2.3f\n",poolF[0],poolF[1]);
  //     printf("%d flow[211,532]: %2.3f,%2.3f\n",seg_idx,poolF[0],poolF[1]);
  // }
  
  // if ((h_idx == 211) and (w_idx == 532)){
  //   downsampled[2*seg_idx] = 13;
  //   downsampled[2*seg_idx+1] = -2;
  //   // poolF[fidx] = 13;
  //   // poolF[fidx+1] = -2;
  // }


}


std::tuple<float*,int*>
run_sp_downsample(float* tensor, int* seg,
                       int nspix, int nbatch, int npix, int nftrs){
  
  
  // -- allocate --
  float* down = (float*)easy_allocate(nbatch*nspix*nftrs,sizeof(float));
  int* counts = (int*)easy_allocate(nbatch*nspix,sizeof(int));
  cudaMemset(down, 0., nftrs*nbatch*nspix*sizeof(float));
  cudaMemset(counts, 0, nbatch*nspix*sizeof(int));

  // -- aggregate --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block,nbatch);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  sp_downsample_kernel<<<BlockPerGrid,ThreadPerBlock>>>
    (tensor, seg, down, counts, nspix, nbatch, npix, nftrs);

  // -- normalize [down = down/counts] --
  int num_block2 = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid2(num_block2,nbatch);
  normz_downsample_kernel<<<BlockPerGrid2,ThreadPerBlock>>>
    (down, counts, nspix, nbatch, nftrs);

  return std::make_tuple(down,counts);

}

float* run_sp_upsample(int* seg, float* down,
                       int nspix, int nbatch, int npix, int nftrs){

  // -- allocate --
  float* pool = (float*)easy_allocate(nbatch*npix*nftrs,sizeof(float));
  // printf("npix,nspix: %d,%d\n",npix,nspix);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // -- scatter --
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlockPerGrid(num_block,nbatch);
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  sp_upsample_kernel<<<BlockPerGrid,ThreadPerBlock>>>
    (pool, seg, down, nspix, nbatch, npix, nftrs);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  return pool;

}


std::tuple<float*,float*,int*>
run_sp_pooling(float* tensor, int* seg, int nspix,
               int nbatch, int height, int width, int nftrs){

  // -- init --
  int npix = height*width;

  // -- downsample --
  auto out_down = run_sp_downsample(tensor, seg, nspix, nbatch,  npix, nftrs);
  float* down = std::get<0>(out_down);
  int* counts = std::get<1>(out_down);
  
  // -- upsample --
  float* pooled = run_sp_upsample(seg, down, nspix, nbatch, npix, nftrs);

  // run_sp_video_upsample<<<BlockPerGrid,ThreadPerBlock>>>
  //   (pooled_ptr, seg_ptr, downsampled_ptr, nspix, nbatch, npix, nftrs);

  return std::make_tuple(pooled,down,counts);
}

