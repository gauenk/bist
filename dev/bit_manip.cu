// -- cpp imports --
#include <stdio.h>
#include <assert.h>
#include <bitset>

// -- cuda --
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuda.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

#define THREADS_PER_BLOCK 512

__global__
void view(uint64_t* comparisons, int nspix){

  // -- get superpixel index --
  int spix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_id>=nspix) return;

  // -- decode --
  uint64_t comparison = comparisons[spix_id];
  uint32_t delta32 = uint32_t(comparison>>32);
  // int candidate_spix = static_cast<int>(comparison);
  int candidate_spix = *reinterpret_cast<int*>(&comparison);
  float delta = *reinterpret_cast<float*>(&delta32);
  // float delta = static_cast<float>(static_cast<uint32_t>(comparison>>32));
  if (spix_id<20){
    printf("spix,candidate_spix,delta: %d,%d,%2.4f\n",spix_id,candidate_spix,delta);
  }

}

void main(){


  int nspix = 10;

  uint64_t* comparisons = (uint64_t*)easy_allocate(nspix,sizeof(uint64_t));
  float val = 100.f;
  uint32_t val1 = *reinterpret_cast<uint32_t*>(&val);
  uint64_t val2 = (uint64_t(val1) << 32) | val1;
  int val3 = *reinterpret_cast<int*>(&val);
  cuMemsetD32((CUdeviceptr)comparisons,val1,nspix*sizeof(uint64_t)/sizeof(uint32_t));

  std::cout << "\nval3: " << val3 << "\n\n" << std::endl;
  // std::bitset<32> bits0(*reinterpret_cast<uint32_t*>(&val));
  std::bitset<32> bits1(val1);
  // std::cout << "\n\n" << val << "\n\n\n\n\n\n\n\n\n\n\n\n\n" << std::endl;
  // std::cout << "\n\n" << bits0 << "\n\n\n\n\n\n\n\n\n\n\n\n\n" << std::endl;
  std::cout << "\n\n" << bits1 << "\n\n\n\n\n\n\n\n\n\n\n\n\n" << std::endl;


  int nblocks_for_spix = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlocksSpix(nblocks_for_spix,nbatch);
  view<<<BlocksSpix,NumThreads>>>(comparisons,nspix);


  // // for (int i = 31; i >= 0; --i) {
  // //   std::cout << ((val >> i) & 1);
  // // }
  // printf("\n");
  // for (int i = 31; i >= 0; --i) {
  //   std::cout << ((val1 >> i) & 1);
  // }
  // std::cout << std::endl;
  // exit(1);
  // cudaMemset(comparisons,val1,2*nspix*sizeof(uint32_t));
  // cudaMemset(comparisons,val2,nspix*sizeof(uint64_t));
  // cudaMemset(comparisons,1,nspix*sizeof(uint64_t));
  // cudaMemset(comparisons,val3,nspix*sizeof(uint64_t));
  // cuMemsetD2D32((CUdeviceptr)comparisons,0,val1,2*nspix*sizeof(uint32_t),1);
  // cuMemsetD32((CUdeviceptr)comparisons,val1,2*nspix*sizeof(uint32_t));

}