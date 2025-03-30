
/*********************************************************************************

     - This code shifts labels according to the input optical flow
     and marks overlapping regions as `invalid' using that
     conveinent 'atomicAdd' features which reads atomicly from memory.

     - this atomic read makes invaliding the overlapping regions _very_ fast
*********************************************************************************/


// -- cpp imports --
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#define THREADS_PER_BLOCK 512

// -- project imports --
#include "structs.h"
#include "init_utils.h"
#include "atomic_helpers.h"


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

     Usually, pix_labels == spix

***********************************************************/

__global__
void shift_labels_kernel(int* pix_labels, int* spix, float* flow, 
                         // unsigned long long* shifted_spix,
                         uint64_t* shifted_spix,
                         int* counts,
                         int* sizes, int* min_sizes,
                         int npix, int nspix, int nbatch,
                         int height, int width){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  int h_idx = pix_idx / width;
  int w_idx = pix_idx % width;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  
  // -- superpixel at sources --
  int spix_label = *(spix+pix_idx);
  assert(spix_label < nspix);
  int pix_label = *(pix_labels+pix_idx);

  // -- superpixel size --
  int spix_size = *(sizes + nspix*batch_idx+spix_label);
  if (spix_size <= 0){
    printf("ERROR [shift_labels.cu] (spix_label,spix_size): (%d,%d)\n",spix_label,spix_size);
  }
  assert(spix_size > 0);

  // -- flow at source --
  int flow_offset = 2*(nspix*batch_idx+spix_label);
  int flow_w = round(*(flow+flow_offset));
  int flow_h = round(*(flow+flow_offset+1));
  // float _flow_w = *(flow+flow_offset);
  // float _flow_h = *(flow+flow_offset+1);
  // int flow_w = round(_flow_w);
  // int flow_h = round(_flow_h);
  // int flow_w = 0;
  // int flow_h = 0;

  // -- dest index --
  int h_dest = h_idx+flow_h;
  int w_dest = w_idx+flow_w;

  // -- info --
  // if ((h_dest = 211) and (w_dest == 532)){
  //   printf("flow[211,532]: %2.3f,%2.3f\n",_flow_h,_flow_w);
  // }

  // -- check boundary; skip if oob --
  bool valid_h = (0<=h_dest) and (h_dest<height);
  bool valid_w = (0<=w_dest) and (w_dest<width);
  bool valid = valid_h and valid_w;
  if (not valid){ return; }

  // -- write to destination --
  int dest_idx = h_dest * width + w_dest;
  assert(dest_idx<npix);
  dest_idx = dest_idx + npix*batch_idx;
  uint64_t* shifted_ptr = shifted_spix+dest_idx;
  int* counts_ptr = counts+dest_idx;
  int* min_size_ptr = min_sizes+dest_idx;

  // -- atomic read from mem --
  // int prev_max = atomicMin(counts_ptr,prev_spix != -1);
  // int prev_max = atomicAdd(counts_ptr,1);
  int prev_max = atomicMax(counts_ptr,1);
  // int prev_spix = atomicMax(shifted_ptr,spix_label);

  // spix_size = 10;
  int prev_min = atomicMin(min_size_ptr,spix_size);
  // printf("%d,%d\n",prev_min,*min_size_ptr);
  assert(prev_min>=0);

  // -- new way of setting so smallest survives --
  // assert(spix_label < 10000);
  // unsigned long long scale = spix_size * 10000; // requires spix_labels < 10000
  // unsigned long long new_label = scale + spix_label;
  // unsigned long long prev_spix = atomicMin(shifted_ptr,new_label);
  uint32_t spix_size_ = *reinterpret_cast<uint32_t*>(&spix_size);
  // uint32_t spix_label_ = *reinterpret_cast<uint32_t*>(&spix_label);
  // atomic_min_update_int(shifted_ptr, spix_size_, spix_label_);
  uint32_t pix_label_ = *reinterpret_cast<uint32_t*>(&pix_label);
  atomic_min_update_int(shifted_ptr, spix_size_, pix_label_);

}


// // Extract size from the combined 64-bit value
// __device__ inline uint32_t extract_size(uint64_t combined) {
//     return uint32_t(combined >> 32);
// }

// // Extract id from the combined 64-bit value
// __device__ inline uint32_t extract_id(uint64_t combined) {
//     return uint32_t(combined);
// }



__global__
void decode_kernel(int* shifted_spix, uint64_t* shifted_tmp,
                   int* counts, int* min_sizes,
                   int npix, int nspix, int nbatch,
                   int height, int width){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  int h_idx = pix_idx / width;
  int w_idx = pix_idx % width;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  
  // -- read the size and the coded spix --
  int min_size = min_sizes[pix_idx];
  uint64_t combined = shifted_tmp[pix_idx];
  // int decoded_spix = int(uint32_t(combined));
  int decoded_spix = *reinterpret_cast<int*>(&combined);
  // printf("%d\n",decoded_spix);
  // unsigned long long shifted_tmp_i = shifted_tmp[pix_idx];
  // unsigned long long e_min_size = min_size *10000;
  // unsigned long long decoded_spix = shifted_tmp_i - e_min_size;
  int count = counts[pix_idx];

  // -- optional error checking [doesn't work with unless labels == spix] --
  // if (count > 0){
  //   if (decoded_spix >= nspix){
  //     printf("ERROR [shift_labels.cu] (decoded_spix,nspix): %d,%d\n",decoded_spix,nspix);
  //   }
  //   assert(decoded_spix < nspix);
  // }

  // -- store the decoded spix --
  shifted_spix[pix_idx] = (int)decoded_spix;
  if (count == 0){
    shifted_spix[pix_idx] = -1;
  }

}



/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

// std::tuple<int*,int*>
int* run_shift_labels(int* pix_labels, int* spix, float* flow, int* sizes,
                      int nspix, int nbatch, int height, int width){


    // -- allocate memory --
    int npix = height*width;
    uint64_t* shifted_tmp = (uint64_t*)easy_allocate(nbatch*npix,sizeof(uint64_t));
    int* shifted_spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
    int* counts = (int*)easy_allocate(nbatch*npix,sizeof(int));
    int* min_sizes = (int*)easy_allocate(nbatch*npix,sizeof(int));
    cudaMemset(counts, 0, nbatch*npix*sizeof(int));
    // cudaMemset(min_sizes, 10000, nbatch*npix*sizeof(int));

    // -- init min sizes --
    // uint64_t* comparisons = (uint64_t*)easy_allocate(nspix,sizeof(uint64_t));
    int init_min_size = 100000;
    uint32_t init_32 = *reinterpret_cast<uint32_t*>(&init_min_size);
    size_t num_32bits = nbatch*npix*sizeof(uint64_t)/sizeof(uint32_t);
    cuMemsetD32((CUdeviceptr)shifted_tmp,init_32,num_32bits);
    num_32bits = nbatch*npix*sizeof(int)/sizeof(uint32_t);
    cuMemsetD32((CUdeviceptr)min_sizes,init_32,num_32bits);

    // -- init value --
    // int32_t large_signed_int = 0x7FFFFFFF; // Maximum 32-bit signed integer
    // int64_t value=static_cast<uint64_t>(large_signed_int);
    // cudaMemset(shifted_tmp, value, nbatch*npix*sizeof(uint64_t));
    // cudaMemset(shifted_tmp, 100, 2*nbatch*npix*sizeof(uint32_t));

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("nspix,npix: %d,%d\n",nspix,npix);

    // -- shift --
    shift_labels_kernel<<<BlocksPixels,NumThreads>>>(pix_labels,spix,flow,
                                                     shifted_tmp,counts,sizes,
                                                     min_sizes,npix,
                                                     nspix,nbatch,height,width);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


    // -- decode --
    decode_kernel<<<BlocksPixels,NumThreads>>>(shifted_spix,shifted_tmp,
                                               counts,min_sizes,npix,nspix,
                                               nbatch,height,width);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


    // -- free --
    // cudaFree(comparisons); // not caught by cuda-memcheck
    cudaFree(min_sizes);
    cudaFree(counts);
    cudaFree(shifted_tmp);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


    return shifted_spix;

}

// void init_shift_labels(py::module &m){
//   m.def("shift_labels", &run_shift_labels,"shift labels");
// }


