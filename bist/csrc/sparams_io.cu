/************************************************************

     "Read" means we go from spix_params* to SuperpixelParams
     "Write" means we go from SuperpixelParams to spix_params* 

*************************************************************/

// #include <stdio.h>
// #include <math.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cmath>

// #ifndef MY_SP_STRUCT
// #define MY_SP_STRUCT
// #include "../bass/share/my_sp_struct.h"
// #endif


// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>


#define THREADS_PER_BLOCK 512

// #include "pch.h"
#include "structs.h"
#include "seg_utils.h"
#include "init_utils.h"
#include "sparams_io.h"
#include "init_sparams.h"


/***********************************************************************

                       Full Superpixels

***********************************************************************/


__host__
SuperpixelParams* get_output_params(spix_params* sp_params,
                                     SuperpixelParams* prior_params,
                                     int* ids,int num_ids, int nspix){

  // -- init params --
  // SuperpixelParams sp_params_py = init_tensor_params(nspix);
  SuperpixelParams* sp_params_py = new SuperpixelParams(nspix);
  // SuperpixelParams* sp_params_py = _sp_params_py;

  // -- fill with prior [as much as possible] --
  fill_params_from_params(sp_params_py,prior_params);

  // -- fill the tensors with sp_params --
  params_to_vectors(sp_params_py,sp_params,ids,num_ids);
  return sp_params_py;
  

}


__host__
SuperpixelParams* get_params_as_vectors(spix_params* sp_params,
                                         int* ids, int num_ids, int nspix){

  // -- init params --
  // SuperpixelParams sp_params_py = init_tensor_params(nspix);
  // SuperpixelParams _sp_params_py(nspix);
  SuperpixelParams* sp_params_py = new SuperpixelParams(nspix);

  // -- fill the tensors with sp_params --
  params_to_vectors(sp_params_py,sp_params,ids,num_ids);

  return sp_params_py;
  
}

__host__
void fill_params_from_params(SuperpixelParams* dest_params,
                             SuperpixelParams* src_params){


  // -- check --
  int size = src_params->ids.size();
  int size_d = dest_params->ids.size();
  assert(size_d >= size);
  
  // -- appearance --
  thrust::copy(src_params->mu_app.begin(),
               src_params->mu_app.end(),dest_params->mu_app.begin());
  thrust::copy(src_params->prior_mu_app.begin(),
               src_params->prior_mu_app.end(),dest_params->prior_mu_app.begin());

  // -- shape --
  thrust::copy(src_params->mu_shape.begin(),
               src_params->mu_shape.end(),dest_params->mu_shape.begin());
  thrust::copy(src_params->sigma_shape.begin(),
               src_params->sigma_shape.end(),dest_params->sigma_shape.begin());
  thrust::copy(src_params->prior_mu_shape.begin(),
               src_params->prior_mu_shape.end(),dest_params->prior_mu_shape.begin());
  thrust::copy(src_params->prior_sigma_shape.begin(),
               src_params->prior_sigma_shape.end(),dest_params->prior_sigma_shape.begin());
  thrust::copy(src_params->logdet_sigma_shape.begin(),
               src_params->logdet_sigma_shape.end(),
               dest_params->logdet_sigma_shape.begin());

  // -- helpers --
  thrust::copy(src_params->counts.begin(),
               src_params->counts.end(),dest_params->counts.begin());
  thrust::copy(src_params->sm_counts.begin(),
               src_params->sm_counts.end(),dest_params->sm_counts.begin());
  thrust::copy(src_params->prior_counts.begin(),
               src_params->prior_counts.end(),dest_params->prior_counts.begin());
  thrust::copy(src_params->ids.begin(),src_params->ids.end(),dest_params->ids.begin());

}



__host__ spix_params* get_vectors_as_params(SuperpixelParams* params,
                                            int sp_size, int npix,
                                            int nspix, int nspix_buffer){
  // -- allocate superpixel params --
  const int sparam_size = sizeof(spix_params);
  spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);
  cudaMemset(sp_params,0,nspix_buffer*sparam_size);
  // init_sp_params(sp_params,sp_size,nspix,nspix_buffer,npix);

  // -- check legal accessing --
  // int num = params.ids.size(0);
  // assert(num <= nspix_buffer); // buffer must be larger (and very probably is)

  // -- fill the tensors with sp_params --
  vectors_to_params(params,sp_params);
  return sp_params;
  
}

__host__ void params_to_vectors(SuperpixelParams* sp_params_py,
                                spix_params* sp_params, int* ids, int num){
  
  /****************************************************

                    Unpack Pointers

  *****************************************************/

  // -- appearance --
  auto mu_app = thrust::raw_pointer_cast(sp_params_py->mu_app.data());
  auto prior_mu_app = thrust::raw_pointer_cast(sp_params_py->prior_mu_app.data());

  // -- shape --
  auto mu_shape = thrust::raw_pointer_cast(sp_params_py->mu_shape.data());
  auto sigma_shape = thrust::raw_pointer_cast(sp_params_py->sigma_shape.data());
  auto logdet_sigma_shape = thrust::raw_pointer_cast(sp_params_py->logdet_sigma_shape.data());
  auto prior_mu_shape = thrust::raw_pointer_cast(sp_params_py->prior_mu_shape.data());
  auto prior_sigma_shape=thrust::raw_pointer_cast(sp_params_py->prior_sigma_shape.data());
  auto sample_sigma_shape=thrust::raw_pointer_cast(sp_params_py->sample_sigma_shape.data());

  // -- misc --
  auto counts = thrust::raw_pointer_cast(sp_params_py->counts.data());
  auto icounts = thrust::raw_pointer_cast(sp_params_py->invalid_counts.data());
  auto sm_counts = thrust::raw_pointer_cast(sp_params_py->sm_counts.data());
  auto prior_counts = thrust::raw_pointer_cast(sp_params_py->prior_counts.data());
  thrust::copy(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num),
               sp_params_py->ids.begin());
  // thrust::copy(ids,ids+num,sp_params_py->ids.begin());
  // auto ids = thrust::raw_pointer_cast(sp_params_py->ids.data());
  // int nspix = sp_params_py->ids.size(0);
  // assert(num <= nspix);
  // printf("num: %d\n",num);

  // -- read from [sp_params] into [mu_app,mu_shape,...] --
  int num_blocks = ceil( double(num) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  read_params<<<nblocks,nthreads>>>(mu_app, prior_mu_app,
                                    mu_shape, sigma_shape, logdet_sigma_shape,
                                    prior_mu_shape, prior_sigma_shape, sample_sigma_shape,
                                    counts, icounts, sm_counts, prior_counts,
                                    sp_params, ids, num);
  
}


__global__ 
void read_params(float* mu_app, float* prior_mu_app,
                 double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                 double* prior_mu_shape, double* prior_sigma_shape,
                 double* sample_sigma_shape,
                 int* counts, float* icounts, int* sm_counts, float* prior_counts, 
                 spix_params* sp_params, int* ids, int num_spix){

    // -- filling superpixel params into image --
    int _ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (_ix >= num_spix) return;  // 'nspix' should be 'spix', as passed by the caller
    // if (ix >= max_spix) return;  // 'nspix' should be 'spix', as passed by the caller
    // int ix = ids[_ix];
    // int sp_index = ids[_ix];
    int ix = _ix;
    int sp_index = _ix;
    if (sp_index < 0){ return; }

    // -- offest memory access [appearance] --
    float* mu_app_ix = mu_app + ix * 3;
    float* prior_mu_app_ix = prior_mu_app + ix * 3;  // Handle prior_mu_app

    // -- offest memory access [shape] --
    double* mu_shape_ix = mu_shape + ix * 2;
    double* sigma_shape_ix = sigma_shape + ix * 3;  // Handle sigma_shape
    float* logsigma_shape_ix = logdet_sigma_shape + ix;  // Already in the code
    double* prior_mu_shape_ix = prior_mu_shape + ix * 2;  // Handle prior_mu_shape
    double* prior_sigma_shape_ix = prior_sigma_shape + ix * 3; // Handle prior_sigma_shape
    double* sample_sigma_shape_ix = sample_sigma_shape + ix * 3; // Handle prior_sigma_shape

    // -- misc --
    int* counts_ix = counts + ix;
    float* icounts_ix = icounts + ix;
    int* sm_counts_ix = sm_counts + ix;  
    float* prior_counts_ix = prior_counts + ix;  
                                                                                          
    // -- read spix --
    auto params_ix = sp_params[sp_index];
    
    /*****************************************************

                    Fill the Params                             

    *****************************************************/

    // -- appearance [est] --
    mu_app_ix[0] = params_ix.mu_app.x;
    mu_app_ix[1] = params_ix.mu_app.y;
    mu_app_ix[2] = params_ix.mu_app.z;
    // -- appearance [prior] --
    prior_mu_app_ix[0] = params_ix.prior_mu_app.x;  // Fill prior_mu_app
    prior_mu_app_ix[1] = params_ix.prior_mu_app.y;
    prior_mu_app_ix[2] = params_ix.prior_mu_app.z;

    // -- shape [est] --
    mu_shape_ix[0] = params_ix.mu_shape.x;
    mu_shape_ix[1] = params_ix.mu_shape.y;
    sigma_shape_ix[0] = params_ix.sigma_shape.x;  // Fill sigma_shape
    sigma_shape_ix[1] = params_ix.sigma_shape.y;
    sigma_shape_ix[2] = params_ix.sigma_shape.z;
    logsigma_shape_ix[0] = params_ix.logdet_sigma_shape;  // Already in the code
    // -- shape [prior] --
    prior_mu_shape_ix[0] = params_ix.prior_mu_shape.x;  // Fill prior_mu_shape
    prior_mu_shape_ix[1] = params_ix.prior_mu_shape.y;                                    
    prior_sigma_shape_ix[0] = params_ix.prior_sigma_shape.x;  // Fill prior_sigma_shape
    prior_sigma_shape_ix[1] = params_ix.prior_sigma_shape.y;
    prior_sigma_shape_ix[2] = params_ix.prior_sigma_shape.z;
    sample_sigma_shape_ix[0] = params_ix.sample_sigma_shape.x;  // Fill sample_sigma_shape
    sample_sigma_shape_ix[1] = params_ix.sample_sigma_shape.y;
    sample_sigma_shape_ix[2] = params_ix.sample_sigma_shape.z;


    // -- check zero prior_sigma_shape
    // if (params_ix.prior_sigma_shape.x < 1e-8){
    //   printf("A ZERO! [%d] %2.3f %2.3f %2.3f\n",ix,params_ix.prior_sigma_shape.x,
    //          params_ix.prior_sigma_shape.y,params_ix.prior_sigma_shape.z);
    // }
    
    // if ((ix==30) or (ix == 42) or (ix == 21)){
    //   printf("DEBUG [%d] %2.3f %2.3f %2.3f | %2.3f %2.3f %2.3f\n",ix,
    //          params_ix.prior_sigma_shape.x,
    //          params_ix.prior_sigma_shape.y,
    //          params_ix.prior_sigma_shape.z,
    //          params_ix.sigma_shape.x,
    //          params_ix.sigma_shape.y,
    //          params_ix.sigma_shape.z);
    // }

    // -- misc --
    // float pc0 = params_ix.prior_count;
    // float pc = 1.*params_ix.count;
    // printf("[cpp->py %d] pc: %2.3f, %2.3f\n",sp_index,pc,pc0);
    counts_ix[0] = params_ix.count;
    icounts_ix[0] = params_ix.icount; // not set here!1
    sm_counts_ix[0] = params_ix.sm_count;
    prior_counts_ix[0] = params_ix.prior_count;

}


__host__ void vectors_to_params(SuperpixelParams* sp_params_py,
                                spix_params* sp_params){
  
  // -- unpack python pointers --
  float* mu_app = thrust::raw_pointer_cast(sp_params_py->mu_app.data());
  float* prior_mu_app = thrust::raw_pointer_cast(sp_params_py->prior_mu_app.data());
  double* mu_shape = thrust::raw_pointer_cast(sp_params_py->mu_shape.data());
  double* sigma_shape = thrust::raw_pointer_cast(sp_params_py->sigma_shape.data());
  float* logdet_sigma_shape = thrust::raw_pointer_cast(sp_params_py->logdet_sigma_shape.data());
  double* prior_mu_shape = thrust::raw_pointer_cast(sp_params_py->prior_mu_shape.data());
  double* prior_sigma_shape = thrust::raw_pointer_cast(sp_params_py->prior_sigma_shape.data());
  double* sample_sigma_shape = thrust::raw_pointer_cast(sp_params_py->sample_sigma_shape.data());  

  int* counts = thrust::raw_pointer_cast(sp_params_py->counts.data());
  float* icounts = thrust::raw_pointer_cast(sp_params_py->invalid_counts.data());
  int* sm_counts = thrust::raw_pointer_cast(sp_params_py->sm_counts.data());
  float* prior_counts = thrust::raw_pointer_cast(sp_params_py->prior_counts.data());

  // auto ids = thrust::raw_pointer_cast(sp_params_py->ids.data());
  auto ids = sp_params_py->ids; // should be {0,1,...,nspix-1}
  int nspix = sp_params_py->ids.size();
  // int nspix = *thrust::max_element(ids.begin(), ids.end())+1;
  // int nspix = thrust::raw_pointer_cast(sp_params_py->ids.size()); // keep this for now
  // remove "ids" when we are more sure we won't want it

  // -- write from [mu_app,mu_shape,...] to [sp_params] --
  int num_blocks = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_params<<<nblocks,nthreads>>>(mu_app, prior_mu_app,
                                     mu_shape,  sigma_shape,  logdet_sigma_shape,
                                     prior_mu_shape,  prior_sigma_shape,
                                     sample_sigma_shape,
                                     counts,  icounts, sm_counts, prior_counts,
                                     sp_params, nspix);

}


__host__
void write_prior_counts(SuperpixelParams* src_params,spix_params* dest_params,
                        int* ids, int nactive){
  float* prior_counts = thrust::raw_pointer_cast(src_params->prior_counts.data());
  int num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  write_prior_counts_kernel<<<nblocks,nthreads>>>(prior_counts, dest_params,
                                                  ids, nactive);
}

__global__
void write_prior_counts_kernel(float* prior_counts, spix_params* sp_params,
                               int* ids, int nactive) {
    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // if (ix >= nspix) return;
    if (ix >= nactive) return;
    int spix_id = ids[ix];
    // printf("[%d]: %d,%f\n",ix,prior_counts[ix],sp_params[ix].prior_count);
    sp_params[spix_id].prior_count = prior_counts[spix_id];
}


__global__
void write_params(float* mu_app, float* prior_mu_app,
                  double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                  double* prior_mu_shape, double* prior_sigma_shape,
                  double* sample_sigma_shape,
                  int* counts, float* icounts, int* sm_counts, float* prior_counts,
                  spix_params* sp_params, int nspix) {

   /**********************************************************************

           Fills sp_params with information from SuperpixelParams

           Option 1.) Always fill all of them.

           The length of any tensor in "SuperpixelParams"
           is the maximum size of the superpixel

    **********************************************************************/

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nspix) return;

    // -- offset memory access for appearance --
    float* mu_app_ix = mu_app + ix * 3;
    float* prior_mu_app_ix = prior_mu_app + ix * 3;

    // -- offset memory access for shape --
    double* mu_shape_ix = mu_shape + ix * 2;
    double* sigma_shape_ix = sigma_shape + ix * 3;
    float* logdet_sigma_shape_ix = logdet_sigma_shape + ix;
    double* prior_mu_shape_ix = prior_mu_shape + ix * 2;
    double* prior_sigma_shape_ix = prior_sigma_shape + ix * 3;
    double* sample_sigma_shape_ix = sample_sigma_shape + ix * 3;

    // -- misc --
    int* counts_ix = counts + ix;
    float* icounts_ix = icounts + ix;
    int* sm_counts_ix = sm_counts + ix;
    float* prior_counts_ix = prior_counts + ix;

    // -- read spix --
    // int sp_index = ids[ix];
    int sp_index = ix;
    if (sp_index < 0) return;

    // -- write params from spix_params into the tensors --

    // -- appearance [est] --
    // sp_params[sp_index].mu_app.x = mu_app_ix[0];
    // sp_params[sp_index].mu_app.y = mu_app_ix[1];
    // sp_params[sp_index].mu_app.z = mu_app_ix[2];
    float3 _mu_app;
    _mu_app.x = mu_app_ix[0];
    _mu_app.y = mu_app_ix[1];
    _mu_app.z = mu_app_ix[2];
    sp_params[sp_index].mu_app = _mu_app;

    // -- appearance [prior] --
    float3 _mu_app_pr;
    _mu_app_pr.x = prior_mu_app_ix[0];
    _mu_app_pr.y = prior_mu_app_ix[1];
    _mu_app_pr.z = prior_mu_app_ix[2];
    sp_params[sp_index].prior_mu_app = _mu_app_pr;

    // -- shape [est] --
    double2 _mu_shape;
    _mu_shape.x = mu_shape_ix[0];
    _mu_shape.y = mu_shape_ix[1];
    // sp_params[sp_index].mu_shape.x = mu_shape_ix[0];
    // sp_params[sp_index].mu_shape.y = mu_shape_ix[1];
    sp_params[sp_index].mu_shape = _mu_shape;
    sp_params[sp_index].sigma_shape.x = sigma_shape_ix[0];
    sp_params[sp_index].sigma_shape.y = sigma_shape_ix[1];
    sp_params[sp_index].sigma_shape.z = sigma_shape_ix[2];
    sp_params[sp_index].logdet_sigma_shape = logdet_sigma_shape_ix[0];

    // -- shape [prior] --
    sp_params[sp_index].prior_mu_shape.x = prior_mu_shape_ix[0];
    sp_params[sp_index].prior_mu_shape.y = prior_mu_shape_ix[1];
    // sp_params[sp_index].prior_sigma_shape.x = prior_sigma_shape_ix[0];
    // sp_params[sp_index].prior_sigma_shape.y = prior_sigma_shape_ix[1];
    // sp_params[sp_index].prior_sigma_shape.z = prior_sigma_shape_ix[2];

    // -- shape ... --
    double3 _pr_cov;
    _pr_cov.x = prior_sigma_shape_ix[0];
    _pr_cov.y = prior_sigma_shape_ix[1];
    _pr_cov.z = prior_sigma_shape_ix[2];
    // float det = cov_xx * cov_yy - cov_xy*cov_xy;
    // if (det < 0){ det = 0.00001; }
    // prior_sigma_shape_ptr[0] = icov_yy/det;
    // prior_sigma_shape_ptr[1] = -icov_xy/det;
    // prior_sigma_shape_ptr[2] = icov_xx/det;
    // sp_params[sp_index].prior_sigma_shape.x = cov_yy/det;
    // sp_params[sp_index].prior_sigma_shape.y = -cov_xy/det;
    // sp_params[sp_index].prior_sigma_shape.z = cov_xx/det;
    sp_params[sp_index].prior_sigma_shape = _pr_cov;
    sp_params[sp_index].prior_icov = invert_cov(_pr_cov);


    double3 _s_cov;
    _s_cov.x = sample_sigma_shape_ix[0];
    _s_cov.y = sample_sigma_shape_ix[1];
    _s_cov.z = sample_sigma_shape_ix[2];
    sp_params[sp_index].sample_sigma_shape = _s_cov;

    // if (sp_index == 100){
    //   printf("[tag!] %2.3lf %2.3lf %2.3lf\n",
    //          sp_params[sp_index].prior_icov.x,
    //          sp_params[sp_index].prior_icov.y,
    //          sp_params[sp_index].prior_icov.z);
    // }
    // sp_params[sp_index].prior_sigma_shape.y = cov_xy;
    // sp_params[sp_index].prior_sigma_shape.z = cov_yy;
    // sp_params[sp_index].prior_icov_eig = get_cov_eigenvals(cov);
    // printf("prior_sigma: %lf %lf %lf\n",
    //        sp_params[sp_index].prior_sigma_shape.x,
    //        sp_params[sp_index].prior_sigma_shape.y,
    //        sp_params[sp_index].prior_sigma_shape.z);

    // if (sp_index==30){
    //   printf("LOAD2C++ [%d] %2.3f %2.3f %2.3f\n",sp_index,
    //          _pr_cov.x,
    //          _pr_cov.y,
    //          _pr_cov.z);
    //   // printf("DEBUG [%d] %2.3f %2.3f %2.3f | %2.3f %2.3f %2.3f\n",ix,
    //   //        params_ix.prior_sigma_shape.x,
    //   //        params_ix.prior_sigma_shape.y,
    //   //        params_ix.prior_sigma_shape.z,
    //   //        params_ix.sigma_shape.x,
    //   //        params_ix.sigma_shape.y,
    //   //        params_ix.sigma_shape.z);
    // }

    // -- misc --
    // float pc0 = prior_counts_ix[0];
    // float pc = sqrt(1.*counts_ix[0]);
    // printf("[py->cpp %d] pc: %2.3f,%2.3f\n",sp_index,pc,pc0);
    // pc = (pc > 0) ? pc : pc0;
    sp_params[sp_index].count = counts_ix[0];
    sp_params[sp_index].icount = icounts_ix[0];
    sp_params[sp_index].sm_count = sm_counts_ix[0];
    sp_params[sp_index].prior_count = prior_counts_ix[0];
    // sp_params[sp_index].prior_count = pc;
    sp_params[sp_index].prop = true;
    sp_params[sp_index].valid = 0; // this is set later

    // sp_params[sp_index].prop = true;
}


__device__
double3 invert_cov(double3 cov) {

  // -- unpack --
  // double x = cov.x;
  // double y = cov.y;
  // double z = cov.z;

  // -- invert --
  double pr_det = cov.x*cov.z - cov.y*cov.y;
  double x = cov.z/pr_det;
  double y = cov.y/pr_det;
  double z = cov.x/pr_det;

  return make_double3(x, y, z);
  // return make_double3(cov.x, cov.y, cov.z);
}


__device__
double3 get_cov_eigenvals(double3 cov) {

  // -- https://en.wikipedia.org/wiki/Eigenvalue_algorithm#2.C3.972_matrices

  // -- unpack --
  double s11 = cov.x;
  double s12 = cov.y;
  double s22 = cov.z;

  // Calculate the trace and determinant
  double determinant = (s11 * s22 - s12 * s12); 
  double trace = (s11 + s22)*determinant;
  //printf("s11,s12,s22,det,trace: %lf %lf %lf %lf %lf\n",s11,s12,s22,determinant,trace);

  // // -- info --
  // printf("sxx,sxy,syy,pc: %lf %lf %lf %f %lf | %d %d | %lld %lld %lld\n",
  //        sigma.x,sigma.y,sigma.z,prior_count,count,
  //        sum.x,sum.y,sq_sum.x,sq_sum.y,sq_sum.z);
  // assert(determinant>0.0001);
  // printf("s11,s22,det,trace: %lf %lf %lf %lf\n",s11,s22,determinant,trace);

  // Calculate the square root term
  double tmp = (trace * trace)/4.;
  double term;
  if (tmp > determinant){
    term = sqrt(tmp - determinant);
  }else{
    term = 0;
  }

  // Compute the two eigenvalues
  double lambda1 = (trace / 2) + term;
  double lambda2 = (trace / 2) - term;
  // printf("det,trace,term: %lf %lf %lf\n",determinant,trace,term);

  return make_double3(lambda1, lambda2, trace);
}



/*********************************************************************





             -=-=-=-=- Python API  -=-=-=-=-=-

    Update the priors using the spix, img, and SuperpixelParams







**********************************************************************/


__global__
void update_prior_kernel(float* mu_app, float* prior_mu_app,
                         double* mu_shape, double* prior_mu_shape,
                         double* sigma_shape, double* prior_sigma_shape,
                         double* sample_sigma_shape,
                         int* counts, int* ids, int nspix, int prev_nspix, bool invert) {

    // -- filling superpixel params into image --
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= nspix) return;
    int sp_index = ids[ix];
    if (sp_index < 0){ return; } // not needed; remove me
    if (counts[sp_index] <= 0){ return; } // skip if never alive
    // printf("[update_prior_kernel.a]: %d\n",sp_index);
    // if (sp_index < prev_nspix){ return; } // big change; allow updates to prior.
    // printf("[update_prior_kernel]: %d\n",sp_index);

    // -- mean appearance --
    float* prior_mu_app_ptr = prior_mu_app + 3*sp_index;
    float* mu_app_ptr = mu_app + 3*sp_index;
    prior_mu_app_ptr[0] = mu_app_ptr[0];
    prior_mu_app_ptr[1] = mu_app_ptr[1];
    prior_mu_app_ptr[2] = mu_app_ptr[2];

    // -- mean shape --
    double* prior_mu_shape_ptr = prior_mu_shape + 2*sp_index;
    double* mu_shape_ptr = mu_shape + 2*sp_index;
    prior_mu_shape_ptr[0] = mu_shape_ptr[0];
    prior_mu_shape_ptr[1] = mu_shape_ptr[1];

    // -- cov shape --
    double* prior_sigma_shape_ptr = prior_sigma_shape + 3*sp_index;
    double* sigma_shape_ptr = sigma_shape + 3*sp_index;
    double icov_xx = sigma_shape_ptr[0];
    double icov_xy = sigma_shape_ptr[1];
    double icov_yy = sigma_shape_ptr[2];
    double det = icov_xx * icov_yy - icov_xy*icov_xy;
    if (det < 0){ det = 0.000001; }

    if (invert){
      prior_sigma_shape_ptr[0] = icov_yy/det;
      prior_sigma_shape_ptr[1] = -icov_xy/det;
      prior_sigma_shape_ptr[2] = icov_xx/det;
    }else{
      prior_sigma_shape_ptr[0] = icov_xx;
      prior_sigma_shape_ptr[1] = icov_xy;
      prior_sigma_shape_ptr[2] = icov_yy;
    }

    // prior_icov_ptr[0] = prior_sigma_shape_ptr[0];
    // prior_icov_ptr[1] = prior_sigma_shape_ptr[1];
    // prior_icov_ptr[2] = prior_sigma_shape_ptr[2];
    // prior_sigma_shape_ptr[0] = sigma_shape_ptr[0];
    // prior_sigma_shape_ptr[1] = sigma_shape_ptr[1];
    // prior_sigma_shape_ptr[2] = sigma_shape_ptr[2];

}


void run_update_prior(SuperpixelParams* params, int* ids,
                      int npix, int nspix, int prev_nspix, bool invert){

    // -- check --
    // CHECK_INPUT(spix);
    // CHECK_INPUT(params.mu_app);
    // CHECK_INPUT(params.mu_shape);
    // CHECK_INPUT(params.sigma_shape);
    // CHECK_INPUT(params.logdet_sigma_shape);
    // CHECK_INPUT(params.counts);
    // CHECK_INPUT(params.prior_counts);

    // -- get spixel parameters as tensors --
    // int max_spix = spix.max().item();
    // auto unique_ids = std::get<0>(at::_unique(spix));
    // auto ids = unique_ids.data();
    // int npix = spix.size();
    // int _nspix = *thrust::max_element(spix.begin(), spix.end())+1;
    // thrust::device_vector<int> uniq_ids = \
    //   get_unique(thrust::raw_pointer_cast(spix.data()), npix);
    // int nspix = uniq_ids.size();
    // int* ids = thrust::raw_pointer_cast(uniq_ids.data());
    // int _nspix = spix.max().item()+1;
    // int _min = spix.min().item();
    // printf("nspix,_nspix,_min: %d,%d,%d\n",nspix,_nspix,_min);
    // assert(nspix <= _nspix);

    // -- unpack --
    float* mu_app = thrust::raw_pointer_cast(params->mu_app.data());
    float* prior_mu_app = thrust::raw_pointer_cast(params->prior_mu_app.data());
    double* mu_shape = thrust::raw_pointer_cast(params->mu_shape.data());
    double* prior_mu_shape = thrust::raw_pointer_cast(params->prior_mu_shape.data());
    double* sigma_shape = thrust::raw_pointer_cast(params->sigma_shape.data());
    double* prior_sigma_shape =thrust::raw_pointer_cast(params->prior_sigma_shape.data());
    double* sample_sigma_shape =thrust::raw_pointer_cast(params->sample_sigma_shape.data());
    int* counts =thrust::raw_pointer_cast(params->counts.data());

    // -- launch copy kernel --
    int num_blocks = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    dim3 nblocks(num_blocks);
    dim3 nthreads(THREADS_PER_BLOCK);
    update_prior_kernel<<<nblocks,nthreads>>>(mu_app, prior_mu_app, mu_shape,
                                              prior_mu_shape, sigma_shape,
                                              prior_sigma_shape, sample_sigma_shape,
                                              counts, ids,nspix, prev_nspix, invert);

}


