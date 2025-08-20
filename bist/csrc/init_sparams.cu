/*************************************************

        This script initializes spix_params
            WITH and WITHOUT propogation

**************************************************/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/extrema.h>

#include "init_sparams.h"
#include "update_params.h"
#define THREADS_PER_BLOCK 512


/*************************************************

      Initialize Empty (but Valid) Superpixels

**************************************************/


// __global__ void viz_params_0(spix_params* params, int nspix_buffer){
//   // assume nspix = number of unique superpixel values per image. NOT necessarily [max_spix value = nspix-1]
//   int spix_id = threadIdx.x + blockIdx.x * blockDim.x;
//   int bi = blockIdx.y;
//   if (spix_id >= 14){ return; }
//   if (spix_id < 0){ return; } // invalid
//   int spix_index = spix_id + nspix_buffer*bi;
//   printf("spix_id: %d, valid?: %d\n",spix_id,params[spix_index].valid);
// }

__host__ void init_sp_params(spix_params* sp_params, float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                              int npix, int nspix, int nspix_buffer,
                             int nbatch, int width, int nftrs, int sp_size){

  // int count = npix/(1.*nspix);
  // //printf("count, sp_size: %d,%d\n",count,sp_size);

  // -- fill sp_params with summary statistics --
  update_params_summ(img, spix, sp_params, sp_helper,
                     prior_sigma_app, npix, nspix_buffer, nbatch, width, nftrs);
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,nbatch);
  //printf("nspix: %d\n", nspix);
  init_sp_params_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sp_params, prior_sigma_app,
                                                         nspix, nspix_buffer,
                                                         npix, sp_size);
}


__host__ void init_sp_params_b(spix_params* sp_params, float prior_sigma_app,
                             float* img, int* spix, spix_helper* sp_helper,
                             thrust::device_vector<int>& nspix, int nspix_buffer, int npix,
                             int nbatch, int width, int nftrs, int sp_size){

  // int count = npix/(1.*nspix);
  // //printf("count, sp_size: %d,%d\n",count,sp_size);

  // -- fill sp_params with summary statistics --
  update_params_summ(img, spix, sp_params, sp_helper,
                     prior_sigma_app, npix, nspix_buffer, nbatch, width, nftrs);

  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,nbatch);
  int* nspix_ptr = thrust::raw_pointer_cast(nspix.data());
  init_sp_params_kernel_b<<<BlockPerGrid,ThreadPerBlock>>>(sp_params, prior_sigma_app,
                                                           nspix_ptr, nspix_buffer, npix, sp_size);
    // {                         
    // int nspix_max = 15;
    // int nthreads = THREADS_PER_BLOCK;
    // int _nblocks = ceil( double(nspix_max) / double(THREADS_PER_BLOCK) );
    // dim3 nblocks(_nblocks,nbatch);
    // viz_params_0<<<nblocks,nthreads>>>(sp_params,nspix_buffer);
    // }
}

__global__ void init_sp_params_kernel(spix_params* sp_params,
                                      float prior_sigma_app,
                                      const int nspix, int nspix_buffer,
                                      int npix, int sp_size){
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;
  int _k = k;
  k = k + nspix_buffer*blockIdx.y; // batch dimension

  /****************************************************

           Shift Summary Statistics to Prior

  *****************************************************/

  int count = npix/(1.*nspix);

  double3 prior_sigma_shape;
  prior_sigma_shape.x = 1.0/sp_size;
  prior_sigma_shape.y = 0.;
  prior_sigma_shape.z = 1.0/sp_size;
  sp_params[k].prior_sigma_shape = prior_sigma_shape;
  // sp_params[k].sm_count = 100;

  // int count = max(sp_params[k].count,1);
  if(_k<nspix) {

    // -- activate! --
    sp_params[k].valid = 1;
    sp_params[k].prior_count = count;

    // -- appearance --
    sp_params[k].prior_mu_app = sp_params[k].mu_app;
    // sp_params[k].prior_sigma_app.x = prior_sigma_app;
    // sp_params[k].prior_sigma_app.y = prior_sigma_app;
    // sp_params[k].prior_sigma_app.z = prior_sigma_app;
    sp_params[k].prior_mu_app_count = 1;
    // sp_params[k].prior_sigma_app_count = count;
    sp_params[k].mu_app.x = 0;
    sp_params[k].mu_app.y = 0;
    sp_params[k].mu_app.z = 0;
    // sp_params[k].sigma_app.x = 0;
    // sp_params[k].sigma_app.y = 0;
    // sp_params[k].sigma_app.z = 0;

    // -- shape --
    // sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    // sp_params[k].prior_mu_shape.x = 0;
    // sp_params[k].prior_mu_shape.y = 0;
    // // sp_params[k].prior_sigma_shape = sp_params[k].sigma_shape;

    // -- prior cov --
    double3 icov;
    icov.x = 1;
    icov.y = 0;
    icov.z = 1;
    sp_params[k].prior_icov = icov;
    // sp_params[sp_index].prior_icov_eig = make_double3(1,1,2);

    // double3 prior_sigma_shape;
    // sp_params[k].prior_sigma_shape = prior_sigma_shape;

    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = count;
    sp_params[k].logdet_prior_sigma_shape = 4*logf(1.*max(count,1));
    sp_params[k].mu_shape.x = 0;
    sp_params[k].mu_shape.y = 0;
    sp_params[k].sigma_shape.x = 0;
    sp_params[k].sigma_shape.y = 0;
    sp_params[k].sigma_shape.z = 0;
    sp_params[k].prop = false;


  }else{
    sp_params[k].count = 0;
    float3 mu_app;
    mu_app.x = 0;
    mu_app.y = 0;
    mu_app.z = 0;
    sp_params[k].mu_app = mu_app;
    sp_params[k].prior_mu_app = mu_app;
    sp_params[k].prior_mu_app_count = 1;

    double2 mu_shape;
    mu_shape.x = 0;
    mu_shape.y = 0;
    sp_params[k].mu_shape = mu_shape;
    sp_params[k].prior_mu_shape_count = 1;

    sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    sp_params[k].prior_mu_shape.x = 0;
    sp_params[k].prior_mu_shape.y = 0;
    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = count;
    sp_params[k].logdet_prior_sigma_shape = 4*logf(1.*max(count,1));
    sp_params[k].mu_shape.x = 0;
    sp_params[k].mu_shape.y = 0;
    sp_params[k].sigma_shape.x = 0;
    sp_params[k].sigma_shape.y = 0;
    sp_params[k].sigma_shape.z = 0;
    sp_params[k].logdet_sigma_shape = 0;

    sp_params[k].prop = false;
    sp_params[k].valid = 0;
    sp_params[k].prior_count = 0;

    // -- prior cov --
    double3 icov;
    icov.x = 1;
    icov.y = 0;
    icov.z = 1;
    sp_params[k].prior_icov = icov;
    // sp_params[sp_index].prior_icov_eig = make_double3(1,1,2);

    // -- fixed for debugging --
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;
    // sp_params[k].prior_sigma_shape.y = 0;
    // sp_params[k].prior_sigma_shape_count = count;
    // sp_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));

  }

}



__global__ void init_sp_params_kernel_b(spix_params* sp_params,
                                      float prior_sigma_app,
                                      const int* nspix, int nspix_buffer,
                                      int npix, int sp_size){
  // the label
  int nspix_b = nspix[blockIdx.y]; // batch dimension
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;
  int _k = k;
  k = k + nspix_buffer*blockIdx.y; // batch dimension

  /****************************************************

           Shift Summary Statistics to Prior

  *****************************************************/

  int count = npix/(1.*nspix_b);

  double3 prior_sigma_shape;
  prior_sigma_shape.x = 1.0/sp_size;
  prior_sigma_shape.y = 0.;
  prior_sigma_shape.z = 1.0/sp_size;
  sp_params[k].prior_sigma_shape = prior_sigma_shape;
  // sp_params[k].sm_count = 100;

  // int count = max(sp_params[k].count,1);
  if(_k<nspix_b) {

    // -- activate! --
    sp_params[k].valid = 1;
    sp_params[k].prior_count = count;

    // -- appearance --
    sp_params[k].prior_mu_app = sp_params[k].mu_app;
    // sp_params[k].prior_sigma_app.x = prior_sigma_app;
    // sp_params[k].prior_sigma_app.y = prior_sigma_app;
    // sp_params[k].prior_sigma_app.z = prior_sigma_app;
    sp_params[k].prior_mu_app_count = 1;
    // sp_params[k].prior_sigma_app_count = count;
    sp_params[k].mu_app.x = 0;
    sp_params[k].mu_app.y = 0;
    sp_params[k].mu_app.z = 0;
    // sp_params[k].sigma_app.x = 0;
    // sp_params[k].sigma_app.y = 0;
    // sp_params[k].sigma_app.z = 0;

    // -- shape --
    // sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    // sp_params[k].prior_mu_shape.x = 0;
    // sp_params[k].prior_mu_shape.y = 0;
    // // sp_params[k].prior_sigma_shape = sp_params[k].sigma_shape;

    // -- prior cov --
    double3 icov;
    icov.x = 1;
    icov.y = 0;
    icov.z = 1;
    sp_params[k].prior_icov = icov;
    // sp_params[sp_index].prior_icov_eig = make_double3(1,1,2);

    // double3 prior_sigma_shape;
    // sp_params[k].prior_sigma_shape = prior_sigma_shape;

    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = count;
    sp_params[k].logdet_prior_sigma_shape = 4*logf(1.*max(count,1));
    sp_params[k].mu_shape.x = 0;
    sp_params[k].mu_shape.y = 0;
    sp_params[k].sigma_shape.x = 0;
    sp_params[k].sigma_shape.y = 0;
    sp_params[k].sigma_shape.z = 0;
    sp_params[k].prop = false;


  }else{
    sp_params[k].count = 0;
    float3 mu_app;
    mu_app.x = 0;
    mu_app.y = 0;
    mu_app.z = 0;
    sp_params[k].mu_app = mu_app;
    sp_params[k].prior_mu_app = mu_app;
    sp_params[k].prior_mu_app_count = 1;

    double2 mu_shape;
    mu_shape.x = 0;
    mu_shape.y = 0;
    sp_params[k].mu_shape = mu_shape;
    sp_params[k].prior_mu_shape_count = 1;

    sp_params[k].prior_mu_shape = sp_params[k].mu_shape;
    sp_params[k].prior_mu_shape.x = 0;
    sp_params[k].prior_mu_shape.y = 0;
    sp_params[k].prior_mu_shape_count = 1;
    sp_params[k].prior_sigma_shape_count = count;
    sp_params[k].logdet_prior_sigma_shape = 4*logf(1.*max(count,1));
    sp_params[k].mu_shape.x = 0;
    sp_params[k].mu_shape.y = 0;
    sp_params[k].sigma_shape.x = 0;
    sp_params[k].sigma_shape.y = 0;
    sp_params[k].sigma_shape.z = 0;
    sp_params[k].logdet_sigma_shape = 0;

    sp_params[k].prop = false;
    sp_params[k].valid = 0;
    sp_params[k].prior_count = 0;

    // -- prior cov --
    double3 icov;
    icov.x = 1;
    icov.y = 0;
    icov.z = 1;
    sp_params[k].prior_icov = icov;
    // sp_params[sp_index].prior_icov_eig = make_double3(1,1,2);

    // -- fixed for debugging --
    // sp_params[k].prior_sigma_shape.x = count*count;
    // sp_params[k].prior_sigma_shape.z = count*count;
    // sp_params[k].prior_sigma_shape.y = 0;
    // sp_params[k].prior_sigma_shape_count = count;
    // sp_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));

  }

}








/*****************************************************
*******************************************************


          Mark active and inactive pixels



*******************************************************
******************************************************/

__global__ void mark_inactive_kernel_b(spix_params* params, int* nspix, int nspix_buffer, int sp_size){
  int spix_id = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = blockIdx.y;
  if (spix_id >= nspix_buffer){ return; }
  int spix_index = spix_id + nspix_buffer*bi;
  // atomicAdd(nvalid,1);
  params[spix_index].valid = 0;
  params[spix_index].prop = false;

  // -- set new prior; assume a contiguous spix --
  if (spix_id >= nspix[bi]){
    double3 prior_sigma_shape;
    prior_sigma_shape.x = 1.0/sp_size;
    prior_sigma_shape.y = 0;
    prior_sigma_shape.z = 1.0/sp_size;
    params[spix_index].prior_sigma_shape = prior_sigma_shape;
  }
}

__global__ void mark_active_kernel_b(spix_params* params, int* ids, int *nspix, int nspix_buffer, int max_nspix){
  // assume nspix = number of unique superpixel values per image. NOT necessarily [max_spix value = nspix-1]
  int id_index = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = blockIdx.y;
  if (id_index >= nspix[bi]){ return; }
  int spix_id = ids[id_index+bi*max_nspix];
  if ((spix_id < 0) || (spix_id >= nspix_buffer)){ return; } // invalid
  // atomicAdd(nvalid,1);
  int spix_index = spix_id + nspix_buffer*bi;
  params[spix_index].valid = 1;
  params[spix_index].prop = true;
  //printf("valid :params[(%d,%d) or %d]\n",bi,spix_id,spix_index);
}

__host__ void mark_active_contiguous_b(spix_params* params, 
                                      thrust::device_vector<int>& nspix,
                                      int nbatch, int nspix_buffer, int sp_size){
  int max_nspix = *thrust::max_element(nspix.begin(), nspix.end());
  thrust::device_vector<int> ids(nbatch*max_nspix);
  // Fill with repeating pattern 0, 1, 2, ..., max_nspix-1
  thrust::transform(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(nbatch * max_nspix),
      ids.begin(),
      [max_nspix] __device__ (int i) { return i % max_nspix; }
  );

  mark_active_b(params, ids, nspix, nbatch, nspix_buffer, sp_size);
}


__host__ void mark_active_b(spix_params* params,
                            thrust::device_vector<int>& ids,
                            thrust::device_vector<int>& nspix,
                            int nbatch, int nspix_buffer, int sp_size){

  // assert(nactive > 0);

  // -- allocate --
  // int nvalid;
  // int* nvalid_gpu;
  // cudaMalloc((void **)&nvalid_gpu, sizeof(int));
  // cudaMemset(nvalid_gpu, 0,sizeof(int));

  // -- pointers --
  int* ids_ptr = thrust::raw_pointer_cast(ids.data());
  int* nspix_ptr = thrust::raw_pointer_cast(nspix.data());

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  //printf("nspix_buffer: %d, sp_size: %d\n",nspix_buffer, sp_size);

  // -- mark activte/inactive --
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  int num_block1 = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 BlockPerGrid1(num_block1,nbatch);
  mark_inactive_kernel_b<<<BlockPerGrid1,ThreadPerBlock>>>(params,nspix_ptr,nspix_buffer,sp_size);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  // -- report--
  //cudaMemcpy(&nvalid, nvalid_gpu, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("[init_sparams.mark_active] ninactive: %d\n",nvalid);
  //cudaMemset(nvalid_gpu, 0,sizeof(int));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  int nspix_max = *thrust::max_element(nspix.begin(), nspix.end());
  //printf("nspix_max: %d\n",nspix_max);
  int num_block2 = ceil( double(nspix_max)/double(THREADS_PER_BLOCK) );
  dim3 BlockPerGrid2(num_block2,nbatch);
  // std::cout << num_block2 <<  " " << nactive << std::endl;
  mark_active_kernel_b<<<BlockPerGrid2,ThreadPerBlock>>>(params,ids_ptr,nspix_ptr,nspix_buffer,nspix_max);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  // -- report--
  //cudaMemcpy(&nvalid, nvalid_gpu, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("[init_sparams] nvalid: %d\n",nvalid);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

  // -- free --
  //cudaFree(nvalid_gpu);

}







/*****************************************************
*******************************************************


          Mark active and inactive pixels



*******************************************************
******************************************************/

__global__ void mark_inactive_kernel(spix_params* params,int nspix_buffer,
                                     int* nvalid, int nspix, int sp_size){
  int spix_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (spix_id >= nspix_buffer){ return; }
  // atomicAdd(nvalid,1);
  params[spix_id].valid = 0;
  params[spix_id].prop = false;

  // -- set new prior --
  // if is beyond our contiguous region of superpixels init to reasonable value
  if (spix_id >= nspix){ 
    double3 prior_sigma_shape;
    prior_sigma_shape.x = 1.0/sp_size;
    prior_sigma_shape.y = 0;
    prior_sigma_shape.z = 1.0/sp_size;
    params[spix_id].prior_sigma_shape = prior_sigma_shape;
  }
}

__global__ void mark_active_kernel(spix_params* params, int* ids,
                                   int nactive, int nspix, int nspix_buffer, int* nvalid){
  int id_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (id_index >= nactive){ return; }
  int spix_id = ids[id_index];
  if ((spix_id < 0) || (spix_id >= nspix_buffer)){ return; } // invalid
  // atomicAdd(nvalid,1);
  params[spix_id].valid = 1;
  params[spix_id].prop = true;

}

__host__ void mark_active_contiguous(spix_params* params, int nspix,
                                     int nspix_buffer, int sp_size){
  thrust::device_vector<int> _grid(nspix);
  thrust::sequence(_grid.begin(), _grid.end(), 0);
  int* ids = thrust::raw_pointer_cast(_grid.data());
  mark_active(params, ids, nspix, nspix, nspix_buffer, sp_size);
}


__host__ void mark_active(spix_params* params, int* ids, int nactive,
                          int nspix, int nspix_buffer, int sp_size){

  assert(nactive > 0);

  // -- allocate --
  int nvalid;
  int* nvalid_gpu;
  cudaMalloc((void **)&nvalid_gpu, sizeof(int));
  cudaMemset(nvalid_gpu, 0,sizeof(int));

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

  // -- mark activte/inactive --
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  int num_block1 = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 BlockPerGrid1(num_block1,1);
  mark_inactive_kernel<<<BlockPerGrid1,ThreadPerBlock>>>(params,nspix_buffer,
                                                         nvalid_gpu,nspix,sp_size);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

  // -- report--
  cudaMemcpy(&nvalid, nvalid_gpu, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("[init_sparams.mark_active] ninactive: %d\n",nvalid);
  cudaMemset(nvalid_gpu, 0,sizeof(int));

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


  int num_block2 = ceil( double(nactive)/double(THREADS_PER_BLOCK) );
  dim3 BlockPerGrid2(num_block2,1);
  // std::cout << num_block2 <<  " " << nactive << std::endl;
  mark_active_kernel<<<BlockPerGrid2,ThreadPerBlock>>>(params,ids,nactive,
                                                       nspix,nspix_buffer,nvalid_gpu);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

  // -- report--
  cudaMemcpy(&nvalid, nvalid_gpu, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("[init_sparams] nvalid: %d\n",nvalid);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

  // -- free --
  cudaFree(nvalid_gpu);

}




/************************************************************


    Initialize Superpixels using Spix from Previous Frame


*************************************************************/

__host__
void init_sp_params_from_past(spix_params* curr_params,
                              spix_params* prev_params,
                              int* curr2prev_map, float4 rescale,
                              int nspix, int nspix_buffer,int npix){
  int num_block = ceil( double(nspix_buffer)/double(THREADS_PER_BLOCK) );
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  init_sp_params_from_past_kernel<<<BlockPerGrid,ThreadPerBlock>>>(curr_params,
                                                                   prev_params,
                                                                   // curr2prev_map,
                                                                   rescale, nspix,
                                                                   nspix_buffer, npix);
}

__global__
void init_sp_params_from_past_kernel(spix_params* curr_params,
                                     spix_params* prev_params,
                                     // int* curr2prev_map,
                                     float4 rescale, int nspix,
                                     int nspix_buffer, int npix){
  // -- ... --
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nspix_buffer) return;
  // int pk = curr2prev_map[k]; // pk = "previous k"
  int pk = k;

  int count = npix/(1.*nspix);
  if(k<nspix) {

    // -- activate! --
    curr_params[k].valid = 1;

    // -- unpack for reading --
    float rescale_mu_app = rescale.x;
    float rescale_sigma_app = rescale.y;
    float rescale_mu_shape = rescale.z;
    float rescale_sigma_shape = rescale.w;

    // -- appearance --
    // int count = prev_params[k].count;
    // curr_params[k].prior_mu_app = prev_params[k].mu_app;
    // curr_params[k].prior_sigma_app.x = 0.002025;
    // curr_params[k].prior_sigma_app.y = 0.002025;
    // curr_params[k].prior_sigma_app.z = 0.002025;
    // curr_params[k].prior_mu_app_count = 1;
    // curr_params[k].prior_sigma_app_count = 1;//count;
    // curr_params[k].mu_app.x = 0;
    // curr_params[k].mu_app.y = 0;
    // curr_params[k].mu_app.z = 0;
    // curr_params[k].sigma_app.x = 0;
    // curr_params[k].sigma_app.y = 0;
    // curr_params[k].sigma_app.z = 0;

    int curr_count = prev_params[pk].count;
    curr_params[k].prior_mu_app = prev_params[pk].mu_app;
    // curr_params[k].prior_sigma_app.x = prev_params[pk].sigma_app.x;
    // curr_params[k].prior_sigma_app.y = prev_params[pk].sigma_app.y;
    // curr_params[k].prior_sigma_app.z = prev_params[pk].sigma_app.z;
    curr_params[k].prior_mu_app_count = max(rescale_mu_app * curr_count,1.0);
    // curr_params[k].prior_sigma_app_count = max(rescale_sigma_app * curr_count,1.0);
    // curr_params[k].prior_mu_app_count = 1;
    // curr_params[k].prior_sigma_app_count = count;
    curr_params[k].mu_app.x = 0;
    curr_params[k].mu_app.y = 0;
    curr_params[k].mu_app.z = 0;
    // curr_params[k].sigma_app.x = 0;
    // curr_params[k].sigma_app.y = 0;
    // curr_params[k].sigma_app.z = 0;

    // -- shape --
    curr_params[k].prior_mu_shape = prev_params[pk].mu_shape;
    // double logdet_shape = prev_params[pk].logdet_sigma_shape;
    // double det = exp(logdet_shape);
    curr_params[k].prior_mu_shape_count = max(rescale_mu_shape * curr_count,1.0);
    curr_params[k].prior_sigma_shape_count = max(rescale_sigma_shape * curr_count,1.0);
    curr_params[k].prior_sigma_shape = prev_params[pk].sigma_shape; // should be inv.
    curr_params[k].logdet_prior_sigma_shape = prev_params[pk].logdet_sigma_shape;

    // curr_params[k].prior_sigma_shape.x = prev_params[pk].sigma_shape.z*det;
    // curr_params[k].prior_sigma_shape.y = -prev_params[pk].sigma_shape.y*det;
    // curr_params[k].prior_sigma_shape.z = prev_params[pk].sigma_shape.x*det;

    // curr_params[k].prior_sigma_shape.x = count*count;
    // curr_params[k].prior_sigma_shape.z = count*count;
    // curr_params[k].prior_sigma_shape.y = 0;
    // curr_params[k].prior_mu_shape_count = 1;//prev_params[pk].count;
    // curr_params[k].prior_sigma_shape_count = count;
    // curr_params[k].logdet_prior_sigma_shape = 4*log(max(count,1));

    curr_params[k].mu_shape.x = 0;
    curr_params[k].mu_shape.y = 0;
    curr_params[k].sigma_shape.x = 0;
    curr_params[k].sigma_shape.y = 0;
    curr_params[k].sigma_shape.z = 0;
    curr_params[k].logdet_sigma_shape = 0;

  }else{
    curr_params[k].count = 0;

    float3 mu_app;
    mu_app.x = 0;
    mu_app.y = 0;
    mu_app.z = 0;
    curr_params[k].mu_app = mu_app;
    curr_params[k].prior_mu_app = mu_app;
    curr_params[k].prior_mu_app_count = 1;

    double2 mu_shape;
    mu_shape.x = 0;
    mu_shape.y = 0;
    curr_params[k].mu_shape = mu_shape;
    curr_params[k].prior_mu_shape_count = 1;

    curr_params[k].valid = 0;
  }


}
