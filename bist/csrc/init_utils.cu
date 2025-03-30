/*************************************************

          This script helps allocate
          and initialize memory for
          supporting information

**************************************************/

#include "init_utils.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#define THREADS_PER_BLOCK 512

/*************************************************

               Allocation

**************************************************/

void throw_on_cuda_error_prop(cudaError_t code){ // new name since two .so objects (ugh)
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}

void* easy_allocate(int size, int esize){
  void* mem;
  try {
    throw_on_cuda_error_prop(cudaMalloc((void**)&mem,size*esize));
  }
  catch (thrust::system_error& e) {
    std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
  }
  return mem;
}

void* easy_allocate_cpu(int size, int esize){
  void* mem;
  mem = malloc(size*esize);
  if (!mem) {
    throw "Malloc Failed.";
  }
  return mem;
}

