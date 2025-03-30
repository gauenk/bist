/*********************************************************************

      Shift and Fill the Superpixel Segmentation using Optical Flow
s
**********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#define THREADS_PER_BLOCK 512

// -- thrust --
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/sequence.h>

// -- [dev only; flow io] opencv --
#include <iostream>
#include <fstream>
// #include <opencv2/core/core.hpp>

// -- [helper] project imports --
#include "structs.h"
#include "init_utils.h"
#include "seg_utils.h"

// -- [primary] project imports --
#include "shift_labels.h"
#include "fill_missing.h"
#include "sp_pooling.h"
#include "split_disconnected.h"
#include "shift_and_fill.h"

struct AddFunctor {
    __host__ __device__
    double operator()(double a, double b) const {
        return a + b;
    }
};


struct MultiplyByScalar {
    float scalar;
    __host__ __device__ MultiplyByScalar(float s) : scalar(s) {}  // Constructor

    __device__ float operator()(float flow_pt) const {
        return flow_pt * scalar;
    }
};

// __global__
// void view_centers(double* centers){
//   int x = threadIdx.x;
//   printf("[%d]: %2.4lf  %2.4lf\n",x,centers[2*x],centers[2*x+1]);
// }

// Custom predicate to check for NaN
struct is_nan {
    __host__ __device__
    bool operator()(float x) const {
        return isnan(x);
    }
};

int run_manual_nan_count(float* arr, int size){
  int count = 0;
  printf("hey\n");
  std::vector<float> vec(arr, arr + size);
  for (float num : vec) {
    if (std::isnan(num)) {
      count += 1;
      // std::cout << "NaN found!" << std::endl;
    // } else {
    //   std::cout << num << std::endl;
    }
    printf(".");
  }
  printf("\n");
  return count;
}

// void save_flow(cv::String fname, float* flow, int height, int width){
//   std::ofstream file1;
//   std::cout << "Writing segmenation to " << fname << std::endl;
//   file1.open(fname);
//   int idx = 0;
//   for (int i = 0; i < height; i++){
//     for (int j = 0; j < width; j++){
//       for (int f = 0; f < 2; f++){
//         if ((j==width-1) and (f == 1)){
//           file1 << flow[idx];
//         }else{
//           file1 << flow[idx] <<",";
//         }
//         idx++;
//       }
//     }
//     file1 << '\n';
//   }
//   file1.close();
// }

// void save_flow_gpu(cv::String fname, float* flow, int height, int width){
//   float* flow_cpu = (float*)malloc(2*height*width*sizeof(float));
//   cudaMemcpy(flow_cpu,flow,2*height*width*sizeof(float),cudaMemcpyDeviceToHost);
//   save_flow(fname, flow_cpu, height, width);
//   free(flow_cpu);
// }



std::tuple<int*,int*>
shift_and_fill(int* spix, SuperpixelParams* params, float* flow,
               int nbatch, int height, int width, bool prop_nc, Logger* logger){


  // -- info --
  int nspix = params->ids.size();
  int npix = height*width;

  // -- log flow shift --
  if (logger != nullptr){
    animate_flow_motion(spix,params,flow,nbatch,height,width,prop_nc,logger);
  }

  // -- pool flow --
  float* flow_gpu = (float*)easy_allocate(2*nbatch*npix,sizeof(float));
  cudaMemcpy(flow_gpu,flow,2*nbatch*npix*sizeof(float),cudaMemcpyHostToDevice);
  // cudaMemset(flow_gpu,0,2*nbatch*npix*sizeof(float));
  auto out_sp_flow = run_sp_downsample(flow_gpu,spix,nspix,nbatch,npix,2);
  float* flow_ds = std::get<0>(out_sp_flow);
  int* counts = std::get<1>(out_sp_flow);
  thrust::device_vector<int> counts_tr(counts,counts+nspix);

  // -- [dev only] --
  // cv::String fname = "flow.csv";
  // save_flow_gpu(fname, flow_gpu, height, width);
  // cv::String fname1 = "flow_ds.csv";
  // save_flow_gpu(fname1, flow_ds, 1, nspix);

  // -- info --
  // float* tmp = run_sp_upsample(spix,flow_ds,nspix,nbatch,npix,2);
  // cudaFree(tmp);
  // int* spix_cpu = (int*)malloc(npix*nbatch*sizeof(int));
  // cudaMemcpy(spix_cpu,spix,nbatch*npix*sizeof(int),cudaMemcpyDeviceToHost);
  // int* counts_cpu = (int*)malloc(nspix*nbatch*sizeof(int));
  // cudaMemcpy(counts_cpu,counts,nbatch*nspix*sizeof(int),cudaMemcpyDeviceToHost);
  // int index = width*211+532;
  // int spix_id = spix_cpu[index];
  // int count = counts_cpu[spix_id];
  // int count_ = params->counts[spix_id];
  // // printf("spix_id,count,count_: %d,%d,%d\n",spix_id,count,count_);

  // -- shift spatial centers --
  thrust::device_ptr<float> flow_f_ptr = thrust::device_pointer_cast(flow_gpu);
  thrust::device_vector<float> flow_f(flow_f_ptr, flow_f_ptr+2*nbatch*npix);
  thrust::device_ptr<float> flow_ds_ptr = thrust::device_pointer_cast(flow_ds);
  thrust::device_vector<float> flow_tr(flow_ds_ptr, flow_ds_ptr+2*nbatch*nspix);
  thrust::device_vector<double> centers = params->mu_shape;
  thrust::transform(centers.begin(),centers.end(),
                    flow_tr.begin(),centers.begin(),AddFunctor());

  // -- check for nans --
  // int nan_count_flowf = thrust::count_if(flow_f.begin(), flow_f.end(), is_nan());
  // int nan_count_flow = thrust::count_if(flow_tr.begin(), flow_tr.end(), is_nan());
  // int nan_count_cen = thrust::count_if(centers.begin(), centers.end(), is_nan());
  // // int nan_count_manual = run_manual_nan_count(flow,2*npix);
  // if ((nan_count_flow > 0) or (nan_count_cen > 0)){
  //   // std::cout << "Number of NaNs: " << nan_count_manual << std::endl;
  //   std::cout << "Number of NaNs: " << nan_count_flowf << std::endl;
  //   std::cout << "Number of NaNs: " << nan_count_flow << std::endl;
  //   std::cout << "Number of NaNs: " << nan_count_cen << std::endl;
  //   exit(1);
  // }

  // -- shift superpixel segmentation --
  // printf("shift\n");
  int* shifted_spix = run_shift_labels(spix, spix, flow_ds, counts,
                                       nspix, nbatch, height, width);
  int* filled_spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
  cudaMemcpy(filled_spix, shifted_spix, nbatch*npix*sizeof(int),cudaMemcpyDeviceToDevice);
  thrust::device_vector<int> _counts_s =  get_spix_counts(filled_spix,1,npix,nspix);
  thrust::host_vector<int> counts_s = _counts_s;
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );


  // dim3 blocks(1);
  // dim3 threads(10);
  // view_centers<<<blocks,threads>>>(thrust::raw_pointer_cast(centers.data()));
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );
  // exit(1);

  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );


  // -- fill --
  // printf("fill0.\n");
  int break_iter = 1000;
  double* centers_ptr = thrust::raw_pointer_cast(centers.data());
  run_fill_missing(filled_spix, centers_ptr, nbatch, height, width, break_iter, logger);

  // -- invalidate disconnectd regions --
  // printf("invalid.\n");
  run_invalidate_disconnected(filled_spix, nbatch, height, width, nspix);

  // -- [testing only ]]
  // int ninvalid = count_invalid(filled_spix,nbatch*npix);
  // printf("num invalid: %d\n",ninvalid);

  // -- fill --
  // printf("fill1.\n");
  run_fill_missing(filled_spix, centers_ptr,
                   nbatch, height, width, break_iter);


  // -- check [testing only] --
  // run_invalidate_disconnected(filled_spix, nbatch, height, width, nspix);
  // int ninvalid = count_invalid(filled_spix,nbatch*npix);
  // // ninvalid = count_invalid(filled_spix,nbatch*npix);
  // assert(ninvalid == 0);


  // -- [dev] --
  // int* counts_cpu = (int*)malloc(nspix*nbatch*sizeof(int));
  // cudaMemcpy(counts_cpu,counts,nbatch*nspix*sizeof(int),cudaMemcpyDeviceToHost);
  thrust::host_vector<float> _pc = params->prior_counts;
  thrust::host_vector<int> smc = params->sm_counts;

  // -- get filled counts --
  thrust::device_vector<int> _counts_f =  get_spix_counts(filled_spix,1,npix,nspix);
  thrust::host_vector<int> counts_f = _counts_f;

  // -- count invalid --
  // invalid_counts = _counts_f - _counts_s; // filled - shifted
  thrust::device_vector<float> invalid_counts(_counts_f.size());
  thrust::transform(_counts_f.begin(), _counts_f.end(), _counts_s.begin(), 
                    invalid_counts.begin(),
                    [] __device__ (int f, int s) {
                      return (f - s) / (static_cast<float>(f)+1e-8);
                    });
  // printf("[saf] counts_s,counts_f @ 131: %d,%d\n",counts_s[131],counts_f[131]);


  // int total_c = 0;
  // int total_s = 0;
  // int total_f = 0;
  // float total_pc = 0;
  // float total_pcc = 0;
  // for(int idx=0; idx<nspix; idx++){

  //   int c_prev = counts_cpu[idx];
  //   int c_shift = counts_s[idx];
  //   int c_fill = counts_f[idx];
  //   float prior_count = _pc[idx];

  //   float delta = c_fill > 0 ? (1.*c_fill)/(1.*c_prev) : 1.;
  //   float pcc = delta*prior_count;

  //   // float interp = (0.2*c_prev)+(0.8*prior_count);
  //   // float interp = prior_count;

  //   // delta = min(delta,2.);
  //   // // if (delta > 1){ delta = delta - 1; }
  //   // delta = abs(1 - delta);
  //   // // // assert(delta >= 0);
  //   // // // assert(delta <=1);
  //   // float alpha = 1 - delta;
  //   // // float interp = (0.5*c_prev)+(0.5*prior_count);
  //   // float interp = (alpha*c_prev)+((1-alpha)*prior_count);

  //   // assert(delta >= 0);
  //   // assert(delta <=1);
  //   // float alpha = 1 - delta;
  //   // // float interp = (0.5*c_prev)+(0.5*prior_count);
  //   // float interp = (alpha*c_prev)+(alpha*prior_count);
  //   // if (delta > 1){
  //   //   float interp = (alpha*c_prev)+((1-alpha)*prior_count);
  //   // }else{
  //   //   float interp = c_prev;
  //   // }
  //   // float interp = c_prev; // tested and good
  //   float interp = c_prev; //?

  //   total_c+=counts_cpu[idx];
  //   total_s+=counts_s[idx];
  //   total_f+=counts_f[idx];
  //   total_pc+=_pc[idx];
  //   total_pcc+=pcc;

  //   // printf("[%d]: %d,%d,%d | %2.2f,%2.2f\n",idx,
  //   //        c_prev,c_shift,c_fill,prior_count,pcc);
  //   // _pc[idx] = c_fill > 0 ? c_fill : prior_count;
  //   // _pc[idx] = c_fill > 0 ? c_fill : prior_count;
  //   // _pc[idx] = c_prev;
  //   // _pc[idx] = c_shift > 0 ? max(c_shift,64) : prior_count;
  //   // _pc[idx] = c_prev > 0 ? max(c_prev,64) : prior_count;
  //   _pc[idx] = interp > 0 ? min(max(interp,100.0),10000.0) : prior_count;
  //   if ((_pc[idx] == 0) and (prior_count == 0)){
  //     printf("zero prior counts @ %d\n",idx);
  //   }

  //   // assert(c_shift <= c_prev);
  //   // assert(c_shift <= c_fill);
  // }

  // printf("%d %d %d |  %2.2f %2.2f | %d\n",total_c,total_s,total_f,
  //        total_pc,total_pcc,params->ids.size());
  // exit(1);
  // -- update pc --
  // thrust::device_vector<float> _pc_gpu = _pc;
  // thrust::copy(_pc.begin(), _pc.end(), params->prior_counts.begin()); // segtrackerv2
  // thrust::device_pointer<float> counts;
  // thrust::copy(_pc.begin(), _pc.end(), params->prior_counts.begin()); // segtrackerv2

  // thrust::transform(thrust::device_pointer_cast(counts),
  //                   thrust::device_pointer_cast(counts) + nspix*nbatch,
  //                   params->prior_counts.begin(),
  //                   thrust::placeholders::_1 * 1.0f);

  // -- set priors --
  // printf("prop_nc: %d\n",prop_nc ? 1 : 0);
  if (prop_nc){
    thrust::transform(thrust::device_pointer_cast(counts),
                      thrust::device_pointer_cast(counts) + nspix*nbatch,
                      _counts_f.begin(),
                      // counts_tr.begin(),
                      // params->prior_counts.begin(),
                      params->prior_counts.begin(),
                      [] __device__ (int curr, float prior) {
                        // return (curr > 0) ? min(max(1.0*curr,100.0),10000.0) : prior;
                        return (curr > 0) ? curr : prior;
                    });
  }

  // thrust::copy(counts.begin(), counts.end(), params->prior_counts.begin());
  thrust::copy(invalid_counts.begin(),invalid_counts.end(),
               params->invalid_counts.begin());


  // thrust::copy(_pc.begin(), _pc.end(), params->prior_counts.begin());
  // prior_counts

  // -- free --
  cudaFree(counts);
  cudaFree(flow_ds);
  cudaFree(flow_gpu);

  // printf("done.\n");
  // return std::make_tuple(filled_spix,perc_invalid);
  // return shifted_spix;
  return std::make_tuple(filled_spix,shifted_spix);


}



/***********************************************

        Logging


***********************************************/

void animate_flow_motion(int* spix, SuperpixelParams* params, float* flow,
                         int nbatch, int height, int width, bool prop_nc, Logger* logger){
  // -- unpack --
  int nspix = params->ids.size();
  int npix = height*width;

  // -- Animate Motion if Logging --
  std::vector<float> shift_perc_list = {0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 1.0};
  for (float shift_perc : shift_perc_list){
    // -- pool flow --
    float* flow_gpu = (float*)easy_allocate(2*nbatch*npix,sizeof(float));
    cudaMemcpy(flow_gpu,flow,2*nbatch*npix*sizeof(float),cudaMemcpyHostToDevice);
    auto flow_gpu_ptr = thrust::device_pointer_cast(flow_gpu);
    thrust::transform(flow_gpu_ptr,flow_gpu_ptr+2*npix*nbatch,flow_gpu_ptr,
                      [shift_perc] __device__ (float flow_pt) { return flow_pt*shift_perc; });
    // thrust::transform(flow_gpu_ptr, flow_gpu_ptr + 2 * npix * nbatch, flow_gpu_ptr,MultiplyByScalar(shift_perc));
    // // thrust::transform(flow_gpu_ptr,flow_gpu_ptr+2*npix*nbatch,flow_gpu_ptr,
    // //                   [] __device__ (float flow_pt) { return flow_pt*shift_perc; });

    // cudaMemset(flow_gpu,0,2*nbatch*npix*sizeof(float));
    auto out_sp_flow = run_sp_downsample(flow_gpu,spix,nspix,nbatch,npix,2);
    float* flow_ds = std::get<0>(out_sp_flow);
    int* counts = std::get<1>(out_sp_flow);
    
    // -- shift spatial centers --
    thrust::device_ptr<float> flow_f_ptr = thrust::device_pointer_cast(flow_gpu);
    thrust::device_vector<float> flow_f(flow_f_ptr, flow_f_ptr+2*nbatch*npix);
    thrust::device_ptr<float> flow_ds_ptr = thrust::device_pointer_cast(flow_ds);
    thrust::device_vector<float> flow_tr(flow_ds_ptr, flow_ds_ptr+2*nbatch*nspix);
    thrust::device_vector<double> centers = params->mu_shape;
    thrust::transform(centers.begin(),centers.end(),
                      flow_tr.begin(),centers.begin(),AddFunctor());
    
    // -- arange npix elements --
    thrust::device_vector<int> pix_labels(npix);
    thrust::sequence(pix_labels.begin(), pix_labels.end());
    int* pix_labels_ptr = thrust::raw_pointer_cast(pix_labels.data());

    // -- shift superpixel segmentation --
    int* shifted_spix = run_shift_labels(pix_labels_ptr, spix, flow_ds, counts, nspix, nbatch, height, width);

    // -- log it! --
    logger->save_shifted_spix(shifted_spix);

    // -- free --
    cudaFree(shifted_spix);
    cudaFree(counts);
    cudaFree(flow_ds);
    cudaFree(flow_gpu);
  }

}



// std::tuple<int*,int*>
// shift_and_fill_log(int* spix, SuperpixelParams* params, float* flow,
//                    int nbatch, int height, int width, bool prop_nc, Logger* logger){


//   // -- info --
//   int nspix = params->ids.size();
//   int npix = height*width;

//   if (logger != nullptr){
//     animate_flow_motion(spix,params,flow,nbatch,height,width,prop_nc,logger);
//   }

//   // -- pool flow --
//   float* flow_gpu = (float*)easy_allocate(2*nbatch*npix,sizeof(float));
//   cudaMemcpy(flow_gpu,flow,2*nbatch*npix*sizeof(float),cudaMemcpyHostToDevice);
//   // auto flow_gpu_ptr = thrust::device_pointer_cast(flow_gpu);
//   // thrust::transform(flow_gpu_ptr,flow_gpu_ptr+2*npix*nbatch,flow_gpu_ptr,
//   //                   [] __device__ (float flow_pt) { return flow_pt*shift_perc; });
//   // cudaMemset(flow_gpu,0,2*nbatch*npix*sizeof(float));
//   auto out_sp_flow = run_sp_downsample(flow_gpu,spix,nspix,nbatch,npix,2);
//   float* flow_ds = std::get<0>(out_sp_flow);
//   int* counts = std::get<1>(out_sp_flow);
  
//   // -- shift spatial centers --
//   thrust::device_ptr<float> flow_f_ptr = thrust::device_pointer_cast(flow_gpu);
//   thrust::device_vector<float> flow_f(flow_f_ptr, flow_f_ptr+2*nbatch*npix);
//   thrust::device_ptr<float> flow_ds_ptr = thrust::device_pointer_cast(flow_ds);
//   thrust::device_vector<float> flow_tr(flow_ds_ptr, flow_ds_ptr+2*nbatch*nspix);
//   thrust::device_vector<double> centers = params->mu_shape;
//   thrust::transform(centers.begin(),centers.end(),
//                     flow_tr.begin(),centers.begin(),AddFunctor());
  
//   // -- shift superpixel segmentation --
//   int* shifted_spix = run_shift_labels(spix, spix, flow_ds, counts, nspix, nbatch, height, width);
//   int* filled_spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
//   cudaMemcpy(filled_spix, shifted_spix, nbatch*npix*sizeof(int),cudaMemcpyDeviceToDevice);
//   thrust::device_vector<int> _counts_s =  get_spix_counts(filled_spix,1,npix,nspix);
//   thrust::host_vector<int> counts_s = _counts_s;

//   // -- fill --
//   // printf("fill0.\n");
//   int break_iter = 1000;
//   double* centers_ptr = thrust::raw_pointer_cast(centers.data());
//   run_fill_missing(filled_spix, centers_ptr,
//                        nbatch, height, width, break_iter, logger);

//   // -- invalidate disconnectd regions --
//   // printf("invalid.\n");
//   run_invalidate_disconnected(filled_spix, nbatch, height, width, nspix);

//   // -- [testing only ]]
//   // int ninvalid = count_invalid(filled_spix,nbatch*npix);
//   // printf("num invalid: %d\n",ninvalid);

//   // -- fill --
//   // printf("fill1.\n");
//   run_fill_missing(filled_spix, centers_ptr,
//                    nbatch, height, width, break_iter);

//   // -- [dev] --
//   thrust::host_vector<float> _pc = params->prior_counts;
//   thrust::host_vector<int> smc = params->sm_counts;

//   // -- get filled counts --
//   thrust::device_vector<int> _counts_f =  get_spix_counts(filled_spix,1,npix,nspix);
//   thrust::host_vector<int> counts_f = _counts_f;

//   // -- count invalid --
//   // invalid_counts = _counts_f - _counts_s; // filled - shifted
//   thrust::device_vector<float> invalid_counts(_counts_f.size());
//   thrust::transform(_counts_f.begin(), _counts_f.end(), _counts_s.begin(), 
//                     invalid_counts.begin(),
//                     [] __device__ (int f, int s) {
//                       return (f - s) / (static_cast<float>(f)+1e-8);
//                     });

//   // -- set priors --
//   if (prop_nc){
//     thrust::transform(thrust::device_pointer_cast(counts),
//                       thrust::device_pointer_cast(counts) + nspix*nbatch,
//                       params->prior_counts.begin(),
//                       params->prior_counts.begin(),
//                       [] __device__ (int curr, float prior) {
//                         return (curr > 0) ? min(max(1.0*curr,100.0),10000.0) : prior;
//                     });
//   }

//   // thrust::copy(counts.begin(), counts.end(), params->prior_counts.begin());
//   thrust::copy(invalid_counts.begin(),invalid_counts.end(),
//                params->invalid_counts.begin());


//   // thrust::copy(_pc.begin(), _pc.end(), params->prior_counts.begin());
//   // prior_counts

//   // -- free --
//   cudaFree(counts);
//   cudaFree(flow_ds);
//   cudaFree(flow_gpu);

//   // printf("done.\n");
//   // return std::make_tuple(filled_spix,perc_invalid);
//   // return shifted_spix;
//   return std::make_tuple(filled_spix,shifted_spix);


// }

