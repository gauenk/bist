


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


// -- project import --
#include "seg_utils.h" // only for view_invalid
#include "structs.h"
#include "init_utils.h"
#include "atomic_helpers.h"
#include "relabel.h"
#define THREADS_PER_BLOCK 512


/*****************************************

      Relabel the Superpixels

******************************************/

/*****************************************

      Pairwise comparison between all
      living superpixels and all dead superpixels....

      the dead superpixels should technically continue shifting somehow
      in the background.... but we won't do this....

      instead, we simply compare the mean app and shape

      [a.] Replace a Propogated&Living Superpixel with a Dead One
           -> if this difference with dead spix is smaller than
              the difference with the previous frame, we relabel it

      [b.] Replace a New&Living Superpixel with a Dead One
          -> if the difference between propogated spatially neighboring superpixels are

      [c.] Replace a Propogated&Living Superpixel with a New One.
          -> threshold on mu

******************************************/

/********************************************************
    Compare...
*********************************************************/
__global__ // identical to other fxn; yes.
void relabel_living_kernel(int* living_ids, 
                           bool* relabel, int* relabel_id, int nliving){

  // -- get pixel index --
  int living_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (living_idx>=nliving) return;

  // -- replace --
  int spix_id = living_ids[living_idx];
  bool is_relabeled = relabel[spix_id];
  int new_spix = relabel_id[spix_id];
  if (not(is_relabeled)){ return; }
  living_ids[living_idx] = new_spix;
}



/********************************************************
    Compare...
*********************************************************/
__global__
void relabel_spix_kernel(int* spix, spix_params* params,
                         bool* relabel, int* relabel_id,
                         int npix, int nspix, int nspix_prev){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;

  // -- replace --
  int spix_id = spix[pix_idx];
  bool is_relabeled = relabel[spix_id];
  int new_spix = relabel_id[spix_id];
  bool is_prop = new_spix < nspix_prev;
  if (not(is_relabeled)){ return; }

  spix[pix_idx] = new_spix;
  params[new_spix].count = params[spix_id].count;
  params[new_spix].icount = params[spix_id].icount;
  // params[spix_id].count = 0;
  params[spix_id].valid=false;
  params[new_spix].valid=true;


  // -- init new spix --
  // if (new_spix >= nspix){
  if (new_spix >= nspix_prev){
    // params[new_spix].prop = is_prop;
    params[new_spix].prop = false;
    params[new_spix].prior_count = params[spix_id].prior_count;
    params[new_spix].prior_sigma_shape = params[spix_id].prior_sigma_shape;
    params[new_spix].sample_sigma_shape = params[spix_id].sample_sigma_shape;
    params[new_spix].prior_mu_app = params[spix_id].prior_mu_app;
    params[new_spix].prior_icov = params[spix_id].prior_icov;
  }

}

/********************************************************
    Compare...
*********************************************************/
__global__
void mark_for_relabel(bool* relabel,
                      int* relabel_id,
                      uint64_t* comparisons,
                      float* ss_comps,
                      bool* is_living, int* max_spix,
                      int nspix, int nspix_prev,
                      float epsilon_reid, float epsilon_new){

  // -- get superpixel index --
  int spix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_id>=nspix) return;

  // -- only keep valid --
  bool spix_is_alive = is_living[spix_id];
  if (not(spix_is_alive)){ return; }

  // -- decode --
  float ss_delta = ss_comps[spix_id];
  // uint32_t ss_delta32 = uint32_t(ss_comp>>32);
  // float ss_delta = *reinterpret_cast<float*>(&ss_delta32);
  // int ss_spix = *reinterpret_cast<int*>(&ss_comp);
  // assert(ss_spix == spix_id);

  // -- decode --
  uint64_t comparison = comparisons[spix_id];
  uint32_t delta32 = uint32_t(comparison>>32);
  int candidate_spix = *reinterpret_cast<int*>(&comparison);
  float delta = *reinterpret_cast<float*>(&delta32);

  // if (spix_id > nspix_prev){
  //   printf("%d,%d,%2.4f\n",spix_id,candidate_spix,delta);
  // }
  // if( (spix_id < nspix_prev) and (candidate_spix != spix_id)){
  //   printf("%d,%d,%2.4f\n",spix_id,candidate_spix,delta);
  // }
  // if( (spix_id < nspix_prev) and (candidate_spix == spix_id)){
  //   printf("%d,%d,%2.4f,%2.4f\n",spix_id,candidate_spix,delta,epsilon_new);
  // }

  // -- dev --
  // if (candidate_spix >= nspix){
  //   printf("Error! [%d]: %d,%2.4f\n",spix_id,candidate_spix,delta);
  // }
  // if (candidate_spix == spix_id){ return; }
  // assert(spix_id < nspix_prev);


  relabel[spix_id] = false;
  relabel_id[spix_id] = -1;
  bool cond_revive = (delta < epsilon_reid);
  cond_revive = cond_revive and (candidate_spix < nspix);
  cond_revive = cond_revive and (candidate_spix != spix_id);
  if (cond_revive){
    bool is_alive = is_living[candidate_spix];
    // assert((is_alive == false) or (candidate_spix == spix_id));
    // -- [a] and [b] --
    relabel[spix_id] = (candidate_spix != spix_id);
    relabel_id[spix_id] = candidate_spix;
    // printf("[relabel] revive: %d -> %d\n",spix_id,candidate_spix);
  }else if (ss_delta > epsilon_new){// and (candidate_spix == spix_id)){
    // -- [c] --
    relabel[spix_id] = true;
    int new_label = atomicAdd(max_spix,1)+1; // indicates a new one.
    relabel_id[spix_id] = new_label;
    // printf("[relabel] new: %d -> %d\n",spix_id,new_label);
  }

}



/********************************************************
    Compare all living superpixels with all dead ones
    and return the one with the smallest difference
*********************************************************/
__global__
void find_most_similar_spix(uint64_t* comparisons,
                            float* ss_comps,
                            spix_params* params,
                            float* mu_app_prior,
                            double* mu_shape_prior,
                            bool* is_living,
                            int height, int width, int nspix,
                            int nspix_prev, int ntotal){

  // -- get reference and query superpixel indices --
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  if (index>=ntotal) return; // reference sshould only check living ones.
  int spix_id = index % nspix;
  int candidate_spix = index / nspix;
  assert(spix_id < nspix);
  assert(candidate_spix < nspix_prev);

  // -- read state --
  bool is_same = candidate_spix == spix_id;
  bool is_spix_alive = is_living[spix_id];
  bool is_alive = is_living[candidate_spix];
  if (is_alive && not(is_same)){ return; } // do NOT check candidate if its alive.
  if (not(is_spix_alive)){ return; }

  // -- compute delta appearance --
  float3 mu_app = params[spix_id].mu_app;
  float* mu_app_p = mu_app_prior + 3*candidate_spix;
  float dx_app = mu_app.x - mu_app_p[0];
  float dy_app = mu_app.y - mu_app_p[1];
  float dz_app = mu_app.z - mu_app_p[2];
  float delta_app = (dx_app*dx_app + dy_app*dy_app + dz_app*dz_app);
  int _count = params[spix_id].count;
    
  // -- compute delta position --
  double2 mu_shape = params[spix_id].mu_shape;
  double* mu_shape_p = mu_shape_prior + 2*candidate_spix;
  double mu_x = mu_shape.x;
  double mu_y = mu_shape.y;
  double dx_shape = (mu_shape.x - mu_shape_p[0])/width;
  double dy_shape = (mu_shape.y - mu_shape_p[1])/height;
  double delta_shape = (dx_shape*dx_shape + dy_shape*dy_shape);
  // if (spix_id == candidate_spix){
  //   printf("[%d,%d,%d,%d] %2.5lf, %2.5lf | %2.5lf, %2.5lf | %2.5lf, %2.5f\n",
  //          spix_id,nspix,nspix_prev,is_alive ? 1 : 0,
  //          mu_shape.x,mu_shape_p[0],mu_shape.y,mu_shape_p[1],
  //          delta_shape,delta_app);
  // }

  // -- total difference --
  // float delta = delta_app + delta_shape/10.f;
  float delta = delta_app + 10.0*delta_shape;
  // if ((delta>10) and (spix_id == candidate_spix)){
  // if ((delta>10) and (spix_id >= nspix_prev)){
  // if ((spix_id >= nspix_prev)){// and (candidate_spix == (nspix_prev-1))){
  //   // printf("%2.3f,(%2.3lf,%2.3lf),(%2.3lf,%2.3lf),(%d,%d),%d,%d\n",
  //   //        delta,mu_shape.x,mu_shape.y,
  //   //        mu_shape_p[0],mu_shape_p[1],
  //   //        width,height,
  //   //        spix_id,candidate_spix);
  //   printf("%2.3f, (%2.3f,%2.3f,%2.3f), (%2.3f,%2.3f,%2.3f), (%d,%d,%d)\n",
  //          delta,mu_app.x,mu_app.y,mu_app.z,
  //          mu_app_p[0],mu_app_p[1],mu_app_p[2],
  //          spix_id,candidate_spix,_count);
  // }
  // printf("delta,cspix: %2.3f,%d\n",delta,candidate_spix);
  // if (( abs(mu_x - 333) < 10) and (abs(mu_y - 41) < 10)){
  //   printf("[tag!] spix_id,cand_id,delta,[x|y|z]: %d %d %2.3f [%2.3f,%2.3f|%2.3f,%2.3f|%2.3f,%2.3f]\n",
  //          spix_id,candidate_spix,delta,mu_app.x,mu_app_p[0],mu_app.y,mu_app_p[1],mu_app.z,mu_app_p[2]);
  // }

  // -- compare --
  // delta = (candidate_spix == 87) ? .0123f : 10.;
  if (is_same){
    ss_comps[spix_id] = delta;
  }
  atomic_min_update_float(comparisons+spix_id,delta,candidate_spix);
}
 

int relabel_spix(int* spix, spix_params* sp_params,
                 SuperpixelParams* params_prev,
                 thrust::device_vector<int>& living_ids,
                 float epsilon_reid, float epsilon_new,
                 int height, int width, int nspix_prev, int _max_spix, Logger* logger){

  // -- log input --
  if (logger != nullptr){
    logger->log_relabel(spix);
  }

  // -- unpack --
  int nbatch = 1;
  float* mu_app_prior = thrust::raw_pointer_cast(params_prev->prior_mu_app.data());
  double* mu_shape_prior = thrust::raw_pointer_cast(params_prev->prior_mu_shape.data());

  // -- init --
  int npix = height*width;
  int nspix = _max_spix + 1;
  int max_spix = _max_spix;
  int nliving = living_ids.size();
  // if (nliving == nspix){ return max_spix; }
  // printf("nliving,nspix: %d,%d\n",nliving,nspix);
  int* max_spix_gpu = (int*)easy_allocate(1,sizeof(int));
  // cudaMemset(max_spix_gpu,0,sizeof(int));
  cudaMemcpy(max_spix_gpu,&max_spix,sizeof(int),cudaMemcpyHostToDevice);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // -- check --
  // printf("a.");
  // view_invalid(sp_params,nspix);

  // -- get living superpixels --
  thrust::device_vector<bool> is_living(nspix,false);
  auto one = thrust::make_constant_iterator(true);
  thrust::scatter(one, one+nliving, living_ids.begin(), is_living.begin());
  bool* is_living_ptr = thrust::raw_pointer_cast(is_living.data());
  int* living_ids_ptr = thrust::raw_pointer_cast(living_ids.data());
  // printf("a.");
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // -- check --
  // printf("b.");
  // view_invalid(sp_params,nspix);

  // -- init comparisons --
  // uint64_t* ss_comps = (uint64_t*)easy_allocate(nspix,sizeof(uint64_t));
  float* ss_comps = (float*)easy_allocate(nspix,sizeof(float));
  uint64_t* comparisons = (uint64_t*)easy_allocate(nspix,sizeof(uint64_t));
  float val = 10000.f;
  uint32_t val_32 = *reinterpret_cast<uint32_t*>(&val);
  cuMemsetD32((CUdeviceptr)comparisons,val_32,nspix*sizeof(uint64_t)/sizeof(uint32_t));

  // -- check --
  // printf("c.");
  // view_invalid(sp_params,nspix);

  // -- pairwise comparison --
  // printf("nspix,nspix_prev: %d,%d\n",nspix,nspix_prev);
  int ntotal = nspix*nspix_prev;
  int nblocks_for_pwd = ceil( double(ntotal) / double(THREADS_PER_BLOCK) ); 
  dim3 BlocksPwd(nblocks_for_pwd,nbatch);
  dim3 NumThreads(THREADS_PER_BLOCK,1);
  // printf("nspix, nspix_prev: %d,%d\n",nspix, nspix_prev);
  find_most_similar_spix<<<BlocksPwd,NumThreads>>>(\
      comparisons,ss_comps,sp_params,mu_app_prior,mu_shape_prior,
      is_living_ptr,height,width,nspix,nspix_prev,ntotal);

  // // printf("b.");
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // -- determine which superpixels are relabeled --
  bool* relabel = (bool*)easy_allocate(nspix,sizeof(bool));
  cudaMemset(relabel,false,nspix*sizeof(bool));
  int* relabel_id = (int*)easy_allocate(nspix,sizeof(int));
  int nblocks_for_spix = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlocksSpix(nblocks_for_spix,nbatch);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // printf("c.");
  mark_for_relabel<<<BlocksSpix,NumThreads>>>(relabel,relabel_id,
                   comparisons,ss_comps,
                   is_living_ptr, max_spix_gpu,
                   nspix, nspix_prev, epsilon_reid, epsilon_new);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );
  // exit(1);
  int old_max_spix = max_spix;
  cudaMemcpy(&max_spix,max_spix_gpu,sizeof(int),cudaMemcpyDeviceToHost);
  int num_new = max_spix - old_max_spix;
  // printf("[summ-relabel]: num_new %d | (old,new | %d,%d)\n",
  //        num_new,old_max_spix,max_spix);

  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );


  // -- relabel the superpixel segmentation --
  int nblocks_for_pix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 BlocksPix(nblocks_for_pix,nbatch);
  // printf("[tag-xyz]: %d %d (old_max,new_max: %d %d)\n",
  //        nspix,nspix_prev,old_max_spix,max_spix);
  relabel_spix_kernel<<<BlocksPix,NumThreads>>>(spix, sp_params,
                                                relabel, relabel_id,
                                                npix, nspix, nspix_prev);

  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // -- update living ids --
  int nblocks_for_nlive = ceil( double(nliving) / double(THREADS_PER_BLOCK) ); 
  dim3 BlocksLive(nblocks_for_nlive,nbatch);
  relabel_living_kernel<<<BlocksLive,NumThreads>>>(living_ids_ptr, relabel,
                                                   relabel_id, nliving);


  // -- logging --
  if (logger != nullptr){
    logger->log_relabel(spix);
  }
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // -- free --
  cudaFree(relabel);
  cudaFree(relabel_id);
  cudaFree(ss_comps);
  cudaFree(comparisons);
  cudaFree(max_spix_gpu);
  
  return max_spix;

}