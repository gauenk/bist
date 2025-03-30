
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
#include <math.h>


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 512
// #define THREADS_PER_BLOCK 256

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "split_merge_prop.h"
#include "seg_utils.h"
// #include "demo_utils.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_p(const float* img, int* seg, int* shifted, bool* border,
                spix_params* sp_params, spix_helper* sp_helper,
                spix_helper_sm_v2* sm_helper,
                int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                float alpha_hastings, float split_alpha,
                float iperc_coeff, float sigma2_app, float sigma2_size, 
                int& count, int idx, int max_spix, 
                const int sp_size, 
                const int npix, const int nbatch,
                const int width, const int height,
                const int nftrs, const int nspix_buffer, Logger* logger){

  // only the propogated spix can be split
  // if(idx%2 == 0){
    count += 1;
    int direction = count%2+1;
    // printf("direction: %d\n",direction);
    // -- run split --
    max_spix = CudaCalcSplitCandidate_p(img, seg, shifted, border,
                                        sp_params, sp_helper, sm_helper,
                                        sm_seg1, sm_seg2, sm_pairs,
                                        sp_size,npix,nbatch,width,height,nftrs,
                                        nspix_buffer, max_spix,
                                        direction, alpha_hastings, split_alpha,
                                        iperc_coeff, sigma2_app, sigma2_size, logger);

  // }
  return max_spix;
}

__host__
void run_merge_p(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm_v2* sm_helper,
               int* sm_seg1, int* sm_seg2, int* sm_pairs,
                 float merge_alpha, float alpha_hastings,
               float sigma2_app, float sigma2_size,
               int& count, int idx, int max_spix,
               const int sp_size, const int npix, const int nbatch,
               const int width, const int height,
                 const int nftrs, const int nspix_buffer, Logger* logger){

  // if( idx%4 == 2){
    // -- run merge --
    // count += 1;
    int direction = count%2;
    // fprintf(stdout,"idx,count,direction: %d,%d,%d\n",idx,count,direction);
    CudaCalcMergeCandidate_p(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                             merge_alpha,sp_size,npix,nbatch,width,height,nftrs,
                           nspix_buffer,direction, alpha_hastings,
                             sigma2_app, sigma2_size, logger);

  // }
}

__host__ void CudaCalcMergeCandidate_p(const float* img, int* seg, bool* border,
                                     spix_params* sp_params,spix_helper* sp_helper,
                                     spix_helper_sm_v2* sm_helper,int* sm_pairs,
                                       float merge_alpha, const int sp_size,
                                     const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float log_alpha,
                                       float sigma2_app, float sigma2_size, Logger* logger){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    float a_0 = 1e4;
    // float i_std = 0.018;
    float i_std = sqrt(sigma2_app)*2;
    float b_0 = i_std * (a_0) ;
    float alpha = exp(log_alpha);

    // -- debug --
    // int nvalid_cpu;
    int* nvalid;
    cudaMalloc((void **)&nvalid, sizeof(int));
    // cudaMemset(nvalid, 0,sizeof(int));

    int nmerges;
    int* nmerges_gpu;
    cudaMalloc((void **)&nmerges_gpu, sizeof(int));
    // cudaMemset(nmerges_gpu, 0,sizeof(int));

    // -- dev debugging --
    // int* they_merge_with_me;
    // cudaMalloc((void **)&they_merge_with_me, sizeof(int)*nspix_buffer);
    // int* i_merge_with_them;
    // cudaMalloc((void **)&i_merge_with_them, sizeof(int)*nspix_buffer);
    // -- --


    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                nspix_buffer, nbatch, height, width,
                                                nftrs, npix, sm_pairs, nvalid);
    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                            sp_params, npix,
                                                            nbatch, width,
                                                            height, direction); 
    if (logger!=nullptr){
      logger->log_merge(seg,sm_pairs,nspix_buffer);
    }
    sum_by_label_merge_p<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);

    // -- summary stats of merge --
    calc_merge_stats_step0_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                             sp_helper,sm_helper,
                                                             nspix_buffer, b_0);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    calc_merge_stats_step1_p<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sm_helper,
                                                             nspix_buffer,a_0, b_0);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    calc_merge_stats_step2_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                               sm_helper,nspix_buffer,
                                                               alpha,merge_alpha);


    // -- update the merge flag: "to merge or not to merge" --
    update_merge_flag_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                          sm_helper,nspix_buffer,
                                                          nmerges_gpu);
    if (logger!=nullptr){
      logger->log_merge_details(sm_pairs,sp_params,sm_helper,nspix_buffer,alpha,merge_alpha);
    }

    // -- count number of merges --
    // cudaMemcpy(&nmerges,nmerges_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // // printf("[merge] nmerges-prop: %d\n",nmerges);
    // cudaMemset(nmerges_gpu, 0,sizeof(int));
    // cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[merge] nvalid: %d\n",nvalid_cpu);
    
    // -- actually merge --
    remove_sp_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                  sm_helper,nspix_buffer,nmerges_gpu);
    // cudaMemcpy(&nmerges,nmerges_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[merge] nmerges-acc: %d\n",nmerges);

    merge_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  

    if (logger!=nullptr){
      logger->log_merge(seg,sm_pairs,nspix_buffer);
    }

    // -- free! --
    cudaFree(nvalid);
    cudaFree(nmerges_gpu);


}


__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg,
                                      int* shifted, bool* border,
                                      spix_params* sp_params,
                                      spix_helper* sp_helper,
                                      spix_helper_sm_v2* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int sp_size, const int npix,
                                      const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_spix,
                                      int direction, float alpha,
                                      float split_alpha, float iperc_coeff,
                                      float sigma2_app, float sigma2_size, Logger* logger){

    if (max_spix>nspix_buffer/2){ return max_spix; }
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float alpha_hasting_ratio =  alpha;
    // float split_alpha = exp(alpha);
    // float a_0 = 1e6;
    // float b_0 = sigma2_app * (a_0) ;
    // float b_0;
    float a_0 = 1e4;
    float i_std = 0.018;
    i_std = sqrt(sigma2_app)*2;
    float b_0 = i_std * (a_0) ;
    int done = 1;
    int* done_gpu;
    int* max_sp;
    int nvalid_cpu;
    int* nvalid;
    cudaMalloc((void **)&nvalid, sizeof(int));
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&done_gpu, sizeof(int)); 

    // -- dev only --
    int* count_rules;
    cudaMalloc((void **)&count_rules, 10*sizeof(int)); 
    cudaMemset(count_rules, 0,10*sizeof(int));
    int* _count_rules = (int*)malloc(10*sizeof(int));

    int distance = 1;
    cudaMemset(sm_seg1, -1, npix*sizeof(int));
    cudaMemset(sm_seg2, -1, npix*sizeof(int));
    cudaMemset(nvalid, 0,sizeof(int));
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


    init_sm_p<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,
                                                sm_helper, nspix_buffer,
                                                nbatch, height, width,
                                                nftrs, npix, sm_pairs, nvalid);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // // printf("[split] nvalid: %d\n",nvalid_cpu);
    // cudaMemset(nvalid, 0,sizeof(int));

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                   sm_helper, nspix_buffer,
                                                   nbatch, width, height, direction,
                                                   seg, max_sp, max_spix);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    init_split_p<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                 sm_helper, nspix_buffer,
                                                 nbatch, width,height, -direction,
                                                 seg, max_sp, max_spix);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );


    // idk what "split_sp" is doing here; init_sm clears the merge fields and
    // so the function returns immediately...
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_spix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    int _dev_count = 0;
    while(done)
    {
        cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(\
                 sm_seg1,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // printf(".\n");

        _dev_count++;
        if(_dev_count>5000){
          gpuErrchk( cudaPeekAtLastError() );
          gpuErrchk( cudaDeviceSynchronize() );
          printf("An error when splitting.\n");

          thrust::device_vector<int> _uniq = get_unique(seg, npix);
          thrust::host_vector<int> uniq = _uniq;
          // Print the vector elements
          for (int i = 0; i < uniq.size(); ++i) {
            std::cout << uniq[i] << " ";
          }
          std::cout << std::endl;
          // cv::String fname = "debug_seg.csv";
          // save_spix_gpu(fname, seg, height, width);
          // fname = "debug_seg1.csv";
          // save_spix_gpu(fname, sm_seg1, height, width);
          exit(1);
        }
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    done = 1;
    distance = 1;
    while(done)
    {
		cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);//?
        calc_split_candidate_p<<<BlockPerGrid,ThreadPerBlock>>>(\
                sm_seg2,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("..\n");
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    // updates the segmentation to the two regions; split either left/right or up/down.
    calc_seg_split_p<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2,
                                                    seg, npix,
                                                    nbatch, max_spix);
    if (logger!=nullptr){
      logger->log_split(sm_seg1,seg);
    }
    

    // -- dev --
    // printf("max_spix: %d\n",max_spix);
    // cv::String fname = "debug_seg.csv";
    // save_spix_gpu(fname, seg, height, width);
    // fname = "debug_seg1.csv";
    // save_spix_gpu(fname, sm_seg1, height, width);
    // fname = "debug_seg2.csv";
    // save_spix_gpu(fname, sm_seg2, height, width);


    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    sum_by_label_split_p<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1,
                                                          shifted, sp_params,
                                                          sm_helper, npix, nbatch,
                                                          height,width,nftrs,max_spix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // calc_bn_split<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs, sp_params, sp_helper,
    //                                                 sm_helper, npix, nbatch, width,
    //                                                 nspix_buffer, b_0, max_spix);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // split_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(img,  sm_pairs,
    //                                                    sp_params,  sp_helper,
    //                                                    sm_helper,
    //                                                    npix, nbatch, width, nftrs,
    //                                                    nspix_buffer, a_0,
    //                                                    b_0, max_spix);

    // -- summary stats --
    calc_split_stats_step0_p<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params, sp_helper,
                                                             sm_helper, nspix_buffer,
                                                             b_0, max_spix);
    calc_split_stats_step1_p<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params, sp_helper,
                                                             sm_helper, nspix_buffer,
                                                             a_0, b_0, max_spix);

    // -- update the flag using hastings ratio --
    update_split_flag_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs, sp_params,
                                                          sm_helper,nspix_buffer,
                                                          alpha, split_alpha,
                                                          iperc_coeff, sp_size,
                                                          max_spix, max_sp);

    // -- do the split --
    split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_spix);
    if (logger!=nullptr){
      logger->log_split(sm_seg1,seg);
    }

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // -- nvalid --
    int prev_max_sp = max_spix;
    cudaMemcpy(&max_spix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nsplits: %d\n",max_spix-prev_max_sp);

    // -- free --
    cudaFree(nvalid);
    cudaFree(max_sp);
    cudaFree(done_gpu);

    return max_spix;

    // sample_estimates_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
    //     sp_params,sm_helper,sp_size,npix,nbatch,width,nspix_buffer,
    //     sigma2_app, sigma2_size, max_spix);

    // // calc_marginal_likelihood<<<BlockPerGrid2,ThreadPerBlock>>>(\
    // //     sp_params,sm_helper,npix,nbatch,width,nspix_buffer,
    // //     sigma2_app, max_spix);

    // split_marginal_likelihood_p<<<BlockPerGrid2,ThreadPerBlock>>>(\
    //     sp_params,sm_helper,sp_size,npix,nbatch,width,nspix_buffer,
    //     sigma2_app, sigma2_size, max_spix,count_rules);

    // // gpuErrchk( cudaPeekAtLastError() );
    // // gpuErrchk( cudaDeviceSynchronize() );

    // // fprintf(stdout,"[s_m.cu] max_spix: %d\n",max_spix);
    // split_hastings_ratio_p<<<BlockPerGrid2,ThreadPerBlock>>>(img, sm_pairs, sp_params,
    //                                                          sp_helper, sm_helper,
    //                                                          npix, nbatch, width, nftrs,
    //                                                          nspix_buffer,sp_size,
    //                                                          alpha_hasting_ratio,
    //                                                          max_spix, max_sp);

    // // -- do the split --
    // split_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
    //                                           sp_params, sm_helper, npix,
    //                                           nbatch, width, height, max_spix);


    // // gpuErrchk( cudaPeekAtLastError() );
    // // gpuErrchk( cudaDeviceSynchronize() );
    // // -- nvalid --
    // int prev_max_sp = max_spix;
    // cudaMemcpy(&max_spix, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nsplits: %d\n",max_spix-prev_max_sp);
    // int nsplits = max_spix-prev_max_sp;

    // // -- dev only --
    // cudaMemcpy(_count_rules, count_rules, 10*sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split-rules]: ");
    // for(int i=0;i<7;i++){
    //   printf("%d ",_count_rules[i]);
    // }
    // printf("| %d\n",nsplits);

    // // if (nsplits>0){
    // //   printf("\n\n\n\n\n\n\n\n\n\n");
    // // }


    // // -- free --
    // free(_count_rules); // dev only
    // cudaFree(count_rules); // dev only
    // cudaFree(nvalid);
    // cudaFree(max_sp);
    // cudaFree(done_gpu);

    // // exit(1);

    // return max_spix;
}



__global__ void init_sm_p(const float* img, const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int nspix_buffer, const int nbatch,
                          const int height, const int width,
                          const int nftrs, const int npix,
                          int* sm_pairs, int* nvalid) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	// if (sp_params[k].valid == 0) return;
    // atomicAdd(nvalid,1); // update valid

	// if (sp_params[k].valid != 0) {
    //   atomicAdd(nvalid,1); // update valid
    // }

    // sm_helper[k].b_n_app.x = 0;
    // sm_helper[k].b_n_app.y = 0;
    // sm_helper[k].b_n_app.z = 0;

    double3 sq_sum_app;
    sq_sum_app.x = 0;
    sq_sum_app.y = 0;
    sq_sum_app.z = 0;
	sm_helper[k].sq_sum_app = sq_sum_app;

    double3 sum_app;
    sum_app.x = 0;
    sum_app.y = 0;
    sum_app.z = 0;
	sm_helper[k].sum_app = sum_app;

    longlong3 sq_sum_shape;
    sq_sum_shape.x = 0;
    sq_sum_shape.y = 0;
    sq_sum_shape.z = 0;
    sm_helper[k].sq_sum_shape = sq_sum_shape;

    longlong2 sum_shape;
    sum_shape.x = 0;
    sum_shape.y = 0;
    sm_helper[k].sum_shape = sum_shape;

    sm_helper[k].count = 0;
    sm_helper[k].hasting = -99999;
    //sp_params[k].count = 0;

    sm_helper[k].merge = false;
    sm_helper[k].remove = false;

    // -- invalidate --
	// if (k>=npix){
    //   printf("WARNING!\n");
    //   return; // skip if more than twice the pixels
    // }
    // assert(k<npix);

    sm_pairs[2*k] = -1;
    sm_pairs[2*k+1] = -1;


    // int k2 = 2*k;
    // if (k2 < 2*npix){
    //   sm_pairs[k2] = -1;
    // }
    // if (k2+1 < 2*npix){
    //   sm_pairs[k2+1] = -1;
    // }

}
// __global__
// void merge_marginal_likelihood_p(int* sm_pairs, spix_params* sp_params,
//                                spix_helper_sm_v2* sm_helper,
//                                const int sp_size,
//                                const int npix, const int nbatch,
//                                const int width, const int nspix_buffer,
//                                float sigma2_app, float sigma2_size){

//     /********************
//            Init
//     **********************/

//     // -- init --
// 	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (k>=nspix_buffer) return;
// 	if (sp_params[k].valid == 0) return;
//     int s = sm_pairs[2*k+1];
//     if (s < 0){ return; }
//     // float count_s = __ldg(&sp_params[s].count);
//     // float count_k = __ldg(&sp_params[k].count);
//     float count_s = __ldg(&sm_helper[s].count);
//     float count_k = __ldg(&sm_helper[k].count);
//     float count_f = count_s + count_k;
//     if((count_f<1)||( count_k<1)||(count_s<1)) return;

//     /********************
  
//           Appearance
   
//     **********************/

//     float prior_count = sp_params[k].prior_count;
//     double3 sum_s = sm_helper[s].sum_app;
//     double3 sum_k = sm_helper[k].sum_app;
//     double3 sum_f;
//     sum_f.x = sum_s.x + sum_k.x;
//     sum_f.y = sum_s.y + sum_k.y;
//     sum_f.z = sum_s.z + sum_k.z;

//     double3 sq_sum_s = sm_helper[s].sq_sum_app;
//     double3 sq_sum_k = sm_helper[k].sq_sum_app;
//     double3 sq_sum_f;
//     sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
//     sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
//     sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

//     double var_app_s;
//     var_app_s = (sq_sum_s.x - sum_s.x*sum_s.x/(1.*count_s));
//     var_app_s += (sq_sum_s.y - sum_s.y*sum_s.y/(1.*count_s));
//     var_app_s += (sq_sum_s.z - sum_s.z*sum_s.z/(1.*count_s));
//     // var_app_s = var_app_s*100;

//     double var_app_k;
//     var_app_k = (sq_sum_k.x - sum_k.x*sum_k.x/(1.*count_k));
//     var_app_k += (sq_sum_k.y - sum_k.y*sum_k.y/(1.*count_k));
//     var_app_k += (sq_sum_k.z - sum_k.z*sum_k.z/(1.*count_k));
//     // var_app_k = var_app_k*100;
//     double var_app = var_app_s + var_app_k;

//     double var_app_f;
//     var_app_f = (sq_sum_f.x - sum_f.x*sum_f.x/(1.*count_f));
//     var_app_f += (sq_sum_f.y - sum_f.y*sum_f.y/(1.*count_f));
//     var_app_f += (sq_sum_f.z - sum_f.z*sum_f.z/(1.*count_f));
//     var_app_f = var_app_f + 1e-8;

//     int mu_x = int(sp_params[k].mu_shape.x);
//     int mu_y = int(sp_params[k].mu_shape.y);



//     // -- appearance --
//     double lprob_s = 0;
//     double lprob_k = 0;
//     // var_app = 0.09;
//     double lprob_f = compare_mu_app_pair(sum_k,sum_s,count_k,count_s)/var_app;
//     // lprob_f = sqrt(lprob_f+1e-10)-0.125; // 0.125 is 10% interval around 0. (z-score)
//     double z_score = sqrt(lprob_f+1e-10);
//     // lprob_f = z_score-0.125;
//     // lprob_f = z_score-0.125/2.0;
//     // lprob_f = z_score - 0.125*1.8;

//     // lprob_f = z_score - 0.125*1.; // just dev
//     // lprob_f = z_score - 0.125*5.; // this looks great.
//     // lprob_f = 0.125*6. - z_score; // if (lf < 0) then no-merge
//     double z_comp = 0.125*1. - z_score; // if (lf > 0) then merge

//     // -- only favor small changes in variance for merge --
//     // double delta_v = 0.0;
//     // delta_v +=abs(var_app_k - var_app_f)/var_app_f;
//     // delta_v += abs(var_app_s - var_app_f)/var_app_f;
//     // delta_v = delta_v;
//     // double b = 2.0;
//     // double var_delta = exp(b*(delta_v-1.))-1.;
//     // lprob_f += var_delta;

//     // -- only favor small changes in variance for merge --
//     // double delta_v = 0.0;
//     double f_stat = (var_app_k<var_app_s) ? var_app_k/var_app_s : var_app_s/var_app_k;
//     double f_comp = 1.26 - f_stat; // if (f_comp > 0) then merge
//     lprob_f = 1.0*((z_comp>0) + (f_comp>0));
//     // lprob_f = z_comp + f_comp;

//     // lprob_f = 100;

//     // if ((abs(mu_x-338) < 10) and (abs(mu_y-57) < 10)){
//     //   printf("[merge_scores:%d,%d] var[s,k]: %2.3lf %2.3lf | z-score: %2.3lf | f-score: %2.3lf | lprob_f: %2.3lf\n",
//     //          k,s,var_app_s,var_app_k,z_score,f_stat,lprob_f);
//     // }


//     // double lprob_s_cond_app = w*compute_l2norm_mu_app_p(sum_s,sum_k,count_s,
//     //                                                     sigma2_prior_var);
//     // double lprob_k = marginal_likelihood_app_p(sum_k,sq_sum_k,count_k,sigma2_app);
//     // double lprob_s = marginal_likelihood_app_p(sum_s,sq_sum_s,count_s,sigma2_app);
//     // double lprob_f = marginal_likelihood_app_p(sum_f,sq_sum_f,count_f,sigma2_app);
//     // double sigma2_prior_var = 1.;
//     // double lprob_k = appearance_variance_p(sum_k,sq_sum_k,count_k,sigma2_prior_var);
//     // double lprob_s = appearance_variance_p(sum_s,sq_sum_s,count_s,sigma2_prior_var);
//     // double lprob_f = appearance_variance_p(sum_f,sq_sum_f,count_f,sigma2_prior_var);

//     // if ((k>100) and (k<120)){
//     //   if (lprob_f < 1.65){
//     //     printf("z-score: %2.5lf\n",lprob_f);
//     //   }else{
//     //     printf("[hey!] z-score: %2.5lf\n",lprob_f);
//     //   }
//     // }


//     // -- gaussian prior --
//     // float _sp_size = 2.*sp_size; // set me to "1" for great results.
//     // float sigma2_gauss = 1.0;
//     // float normz = sqrt(_sp_size);
//     // lprob_f += size_likelihood_p(count_f,_sp_size,sigma2_gauss)/normz;
//     // lprob_s += size_likelihood_p(count_s,_sp_size,sigma2_gauss)/normz;
//     // lprob_k += size_likelihood_p(count_k,_sp_size,sigma2_gauss)/normz;

//     // float sigma2_gauss = 0.01;
//     // float sigma2_gauss = 1.;
//     // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size,sigma2_gauss)/normz;
//     // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size,sigma2_gauss)/normz;


//     // -- include size term --
//     // // int sp_size2 = sp_size*sp_size;
//     // lprob_k += size_likelihood_p(count_k,sp_size,sigma2_size);
//     // lprob_s += size_likelihood_p(count_s,sp_size,sigma2_size);
//     // lprob_f += size_likelihood_p(count_f,sp_size,sigma2_size);

//     // -- include size term --
//     // lprob_k += size_beta_likelihood_p(count_k,sp_size,sigma2_size,npix);
//     // lprob_s += size_beta_likelihood_p(count_s,sp_size,sigma2_size,npix);
//     // lprob_f += size_beta_likelihood_p(count_f,sp_size,sigma2_size,npix);

//     // -- write --
//     sm_helper[k].lprob_f_app = lprob_f;
//     sm_helper[s].lprob_s_cond_app = lprob_s;
//     sm_helper[k].lprob_k_cond_app = lprob_k;

// }

// __global__
// void merge_hastings_ratio_p(const float* img, int* sm_pairs,
//                             spix_params* sp_params,
//                             spix_helper* sp_helper,
//                             spix_helper_sm_v2* sm_helper,
//                             const int npix, const int nbatch,
//                             const int width, const int nftrs,
//                             const int nspix_buffer,
//                             float log_alpha, float merge_offset, int* nmerges) {

// 	// getting the index of the pixel
// 	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (k>=nspix_buffer) return;
// 	if (sp_params[k].valid == 0) return;
//     int s = sm_pairs[2*k+1];
//     if(s<0) return;
//     // atomicAdd(nmerges,1);
//     // printf("k,s: %d,%d\n",k,s);
//     // if (sp_params[k].prop || sp_params[s].prop) { return; } // only non-prop merge?
//     if (sp_params[k].prop and sp_params[s].prop) { return; } 
//     // if(s<=0) return;
//     // atomicAdd(nmerges,1);

//     // -- unpack --
//     // float count_s = __ldg(&sp_params[s].count);
//     // float count_k = __ldg(&sp_params[k].count);
//     float count_k = __ldg(&sm_helper[k].count);
//     float count_s = __ldg(&sm_helper[s].count);
//     float count_f = count_s + count_k;
//     if((count_f<1)||(count_k<1)||(count_s<1)) return;

//     float lprob_k = __ldg(&sm_helper[k].lprob_k_cond_app);
//     float lprob_s = __ldg(&sm_helper[s].lprob_s_cond_app);
//     float lprob_f = __ldg(&sm_helper[k].lprob_f_app) + merge_offset;
//     // float lprob_k = __ldg(&sm_helper[k].numerator_app);
//     // float lprob_s = __ldg(&sm_helper[s].numerator_app);
//     // float lprob_f = __ldg(&sm_helper[k].numerator_f_app) - merge_delta_offset;
//     // float lprob_k = 0;
//     // float lprob_s = 0;
//     // float lprob_f = 0;

//     // -- compute hastings --
//     double alpha = exp(log_alpha);
//     // double log_const = lgammaf(count_f) + lgammaf(alpha) \
//     //   + lgammaf(alpha / 2 + count_k) + lgammaf(alpha / 2 + count_s)\
//     //   - lgammaf(count_s) -lgammaf(count_k)-lgammaf(alpha+count_f)-2*lgamma(alpha/2);
//     double log_const = 0;
//     double hastings = log_const + lprob_f - lprob_k - lprob_s - log_alpha;

//     // if ((k > 1620) and (k < 1720)){
//     //   printf("lprob [f,s,k]: %2.3f %2.3f %2.3f | %2.3lf\n",
//     //          lprob_f,lprob_s,lprob_k,hastings);
//     // }

//     // -- hard limits --
//     // if (count_f > 62500){ // just too big (100x100); always split
//     //   hastings = 1;
//     if (count_f > 62500){ // just too big (100x100); never merge
//       hastings = -1;
//     }else if(count_f < 25){
//       hastings = 1;
//     }

//     // hastings = 1.0;
//     // double hastings = lprob_f - lprob_k - lprob_s - log_alpha;
//     sm_helper[k].hasting = hastings;
//     // sm_helper[k].merge = hastings > 0;
//     // sm_helper[s].merge = hastings > 0;

//     // printf("info[%d,%d] %f,%f,%f|%lf,%f,%f,%f,%lf|\n",k,s,
//     //        count_s,count_k,count_f,
//     //        log_const,lprob_f,lprob_k,lprob_s,hastings);

//     // -- Check hastings and update --
//     return;
//     if(hastings > 0){

//       // atomicAdd(nmerges,1);
//       // printf("info[%d,%d] %f,%f,%f|%lf,%f,%f,%f,%lf|\n",k,s,
//       //        count_s,count_k,count_f,
//       //        log_const,lprob_f,lprob_k,lprob_s,hastings);

//       // printf("info[%d,%d] %lf,%f,%f,%f\n",k,s,log_const,lprob_f,lprob_k,lprob_s);
//       // atomicAdd(nmerges,1);
//       int curr_max = atomicMax(&sm_pairs[2*s],k);
//       if( curr_max == -1){
//         // atomicAdd(nmerges,1);
//         sm_helper[k].merge = true;
//       }
//     }
//     return;
// }




// __global__
// void sample_estimates_p(spix_params* sp_params,
//                         spix_helper_sm_v2* sm_helper,
//                         const int sp_size,
//                         const int npix, const int nbatch,
//                         const int width, const int nspix_buffer,
//                         float sigma2_app, float sigma2_size, int max_spix){

//     /********************
//            Init
//     **********************/

//     // -- init --
// 	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (k>=nspix_buffer) return;
// 	if (sp_params[k].valid == 0) return;

//     // -- split --
//     int s = k + (max_spix+1);
// 	if (s>=nspix_buffer) return;
//     // int count_f = __ldg(&sp_params[k].count);
//     int count_k = __ldg(&sm_helper[k].count);
//     int count_s = __ldg(&sm_helper[s].count);
//     int count_f = count_k + count_s;
//     if((count_f<1)||( count_k<1)||(count_s<1)) return;
//     // count_f = count_k + count_s;

//     // -- shape info --
//     // note! using "prior_count/2" worked well.
//     float prior_count = sp_params[k].prior_count;
//     double3 sigma_k = compute_sigma_shape(sm_helper[k].sum_shape,
//                                           sm_helper[k].sq_sum_shape,
//                                           count_k,prior_count,sp_size);
//     double3 sigma_s = compute_sigma_shape(sm_helper[s].sum_shape,
//                                           sm_helper[s].sq_sum_shape,
//                                           count_s,prior_count,sp_size);
//     longlong2 sum_shape_f = get_sum_shape(sm_helper[s].sum_shape,sm_helper[k].sum_shape);
//     longlong3 sq_sum_shape_f = get_sq_sum_shape(sm_helper[s].sq_sum_shape,
//                                                 sm_helper[k].sq_sum_shape);
//     double3 sigma_f = compute_sigma_shape(sum_shape_f,sq_sum_shape_f,
//                                           count_f,prior_count,sp_size);
//     // if ((k>100) and (k<110)){
//     //   printf("[tag:%d] [(x,y,z): %2.3lf %2.3lf %2.3lf] %2.2f %d\n",
//     //          k,sigma_f.x,sigma_f.x,sigma_f.x,prior_count,sp_size);
//     // }

//     // -- save shape --
//     sm_helper[k].sigma_k = sigma_k;
//     sm_helper[k].sigma_s = sigma_s;
//     sm_helper[k].sigma_f = sigma_f;

// }

__global__
void split_marginal_likelihood_p(spix_params* sp_params,
                                 spix_helper_sm_v2* sm_helper,
                                 const int sp_size,
                                 const int npix, const int nbatch,
                                 const int width, const int nspix_buffer,
                                 float sigma2_app, float sigma2_size,
                                 int max_spix, int* count_rules){

    /********************
           Init
    **********************/

    // -- init --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
	// if (sp_params[k].prop == false) return;    

    // -- split --
    int s = k + (max_spix+1);
	if (s>=nspix_buffer) return;
    int _count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);
    int count_f = count_k + count_s;
    if((count_f<1)||( count_k<1)||(count_s<1)) return;
    if (_count_f != count_f){
      printf("[%d]: %d %d %d %d\n",k,count_f,_count_f,count_k,count_s);
    }
    assert(_count_f == count_f);


    /********************
  
          Appearance
   
    **********************/

    // -- read invalid --
    // double2 mu_shape = sp_params[k].mu_shape;
    // double mu_s_x = mu_shape.x;
    // double mu_s_y = mu_shape.y;
    bool is_prop = sp_params[k].prop;
    int ninvalid_k = sm_helper[k].ninvalid;
    int ninvalid_s = sm_helper[s].ninvalid;
    int new_side = (ninvalid_s > ninvalid_k) ? s : k;
    int old_side = (new_side == k) ? s : k;
    float iperc_k = ninvalid_k / (1.*count_k);
    float iperc_s = ninvalid_s / (1.*count_s);
    float iperc = (new_side == k) ? iperc_k : iperc_s;
    sm_helper[k].lprob_k_cond_app = (new_side == k) ? -1 : 0;
    sm_helper[k].lprob_s_cond_app = (new_side == s) ? -1 : 0;

    // printf("ninvalid [%d,%d]: [%d,%d] [%d,%d]\n",
    //        k,s,ninvalid_k,ninvalid_s,count_k,count_s);

    // -- unpack --
    float prop = sp_params[k].prop;
    float prior_count = sp_params[k].prior_count;
    float3 _mu_prior = sp_params[k].prior_mu_app;
    double3 mu_prior;
    mu_prior.x = _mu_prior.x;
    mu_prior.y = _mu_prior.y;
    mu_prior.z = _mu_prior.z;


    // float3 mu_pr_k = sp_params[k].prior_mu_app;
    // float3 mu_pr_f = mu_pr_k;
    // sp_params[s].prior_mu_app.x = 0;
    // sp_params[s].prior_mu_app.y = 0;
    // sp_params[s].prior_mu_app.z = 0;
    // float3 mu_pr_s = sp_params[s].prior_mu_app;

    // sp_params[s].prior_mu_app_count = 1;
    // int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    // int prior_mu_app_count_k = sp_params[k].prior_mu_app_count;
    // int prior_mu_app_count_f = prior_mu_app_count_k;

    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = sum_s.x + sum_k.x;
    sum_f.y = sum_s.y + sum_k.y;
    sum_f.z = sum_s.z + sum_k.z;

    double3 mu_s = sum_s;
    mu_s.x =mu_s.x/count_s;
    mu_s.y =mu_s.y/count_s;
    mu_s.z =mu_s.z/count_s;

    double3 mu_k = sum_k;
    mu_k.x =mu_k.x/count_k;
    mu_k.y =mu_k.y/count_k;
    mu_k.z =mu_k.z/count_k;

    double3 mu_f = sum_f;
    mu_f.x =mu_f.x/count_f;
    mu_f.y =mu_f.y/count_f;
    mu_f.z =mu_f.z/count_f;

    double3 mu_new = (new_side == k) ? mu_k : mu_s;
    double3 mu_old = (old_side == k) ? mu_k : mu_s;
    float3 mu_pr_k = sp_params[k].prior_mu_app;
    double delta_new = compare_mu_pair(mu_new,mu_prior);
    double delta_old = compare_mu_pair(mu_old,mu_prior);
    double delta_pair = compare_mu_pair(mu_new,mu_old);

    // -- variance --
    double3 sq_sum_s = sm_helper[s].sq_sum_app;
    double3 sq_sum_k = sm_helper[k].sq_sum_app;
    double3 sq_sum_f;
    sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    double var_app_s;
    var_app_s = (sq_sum_s.x - sum_s.x*sum_s.x/(1.*count_s));
    var_app_s += (sq_sum_s.y - sum_s.y*sum_s.y/(1.*count_s));
    var_app_s += (sq_sum_s.z - sum_s.z*sum_s.z/(1.*count_s));
    var_app_s = var_app_s/(3.0*count_s);

    double var_app_k;
    var_app_k = (sq_sum_k.x - sum_k.x*sum_k.x/(1.*count_k));
    var_app_k += (sq_sum_k.y - sum_k.y*sum_k.y/(1.*count_k));
    var_app_k += (sq_sum_k.z - sum_k.z*sum_k.z/(1.*count_k));
    var_app_k = var_app_k/(3.0*count_k);

    double var_app_f;
    var_app_f = (sq_sum_f.x - sum_f.x*sum_f.x/(1.*count_f));
    var_app_f += (sq_sum_f.y - sum_f.y*sum_f.y/(1.*count_f));
    var_app_f += (sq_sum_f.z - sum_f.z*sum_f.z/(1.*count_f));
    var_app_f = var_app_f/(3.0*count_f);
    double var_app_new = (new_side == k) ? var_app_k : var_app_s;
    double var_app_old = (old_side == k) ? var_app_k : var_app_s;

    // -- [shape] write --
    sm_helper[k].lprob_f_shape = 0;
    sm_helper[k].lprob_s_cond_shape = 0;
    sm_helper[k].lprob_s_ucond_shape = 0;
    sm_helper[k].lprob_k_cond_shape = 0;
    sm_helper[k].lprob_k_ucond_shape = 0;

    // -- [app] write --
    sm_helper[k].lprob_s_cond_app = 0;
    sm_helper[k].lprob_k_cond_app = 0;
    sm_helper[k].lprob_f_app = 0;
    sm_helper[k].lprob_s_cond_app = 0;
    sm_helper[k].lprob_k_cond_app = 0;
    sm_helper[k].lprob_f_app = 0;


    /**********************************

           Control Terms Here!

    ***********************************/

    // bool verbose =  (abs(mu_s_x-821)<10) and ((mu_s_y-15)<10);
    float std_for_ztest = sqrt((var_app_s + var_app_k)/3.0);
    float pair_zstat = delta_pair/std_for_ztest;
    float sp2 = 1.0*sp_size*sp_size;
    float iprob = iperc;
    float size_ratio  = count_f / sp2;
    float lsize_ratio = (size_ratio > 1) ? log(size_ratio) : 0.;
    // float var_ratio = var_app_f / (0.004);
    // float var_ratio = (var_app_f-0.0001) / (0.0004) * 10000.;
    // float var_ratio = (var_app_f-0.002)/0.002 * 1.0;
    float var_ratio = (var_app_f-0.002)/0.002 * 1.0;
    var_ratio = (var_ratio > 0) ? var_ratio : 0.0001;
    float esize_ratio = exp(1.0*size_ratio)/exp(1.0);
    // float esize_ratio = exp(100.*var_ratio*size_ratio)/exp(1.0);
    // if (verbose){
    //   printf("[verbose-split %d@(%d,%d)]: %2.3lf %2.3f\n",k,int(mu_s_x),int(mu_s_y),
    //          var_app_f,var_ratio);
    // }


    /**********************************

           RULES BEGIN HERE!

    ***********************************/

    // -- [6] don't split if too small --
    if(count_f < 100){ // just too small (10x10); never split
      sm_helper[k].lprob_f_app = 10;
      atomicAdd(&count_rules[6],1);
      return;
    }

    // -- [0] split if too big or don't split if too small --
    if (count_f > 62500){ // just too big (100x100); always split
      sm_helper[k].lprob_f_app = -10;
      atomicAdd(&count_rules[0],1);
      return;
    }

    // sm_helper[k].lprob_f_app = 10;
    // atomicAdd(&count_rules[0],1);
    // return;

    // -- [0] don't split if variance is small --
    if(var_app_f < 0.0001){
      sm_helper[k].lprob_f_app = 10;
      atomicAdd(&count_rules[6],1);
      return;
    }

    // if (esize_ratio * var_ratio > 12.){ // -- [0] split if too big --
    // if (esize_ratio * var_ratio > 1.){ // -- [0] split if too big --
    if (esize_ratio * var_ratio > 2.0){ // -- [0] split if too big --
      sm_helper[k].lprob_f_app = -10;
      atomicAdd(&count_rules[0],1);
      return;
    }      

    // -- [1] no split if not enough invalid --
    float coeff1 = 1.0/size_ratio;
    float min1 = 0.00;
    float max1 = 0.20;
    float thresh1 = min(max(0.10*coeff1,min1),max1);
    if (iperc < 0.10){
      sm_helper[k].lprob_f_app = 10;
      atomicAdd(&count_rules[1],1);
      return;
    }

    // -- [2] split if means are different --
    float coeff2 = 1.0/size_ratio;
    float min2 = 0.0001;
    float max2 = 0.01;
    float thresh2_z = min(max(0.05*coeff2,min2),max2);
    float thresh2 = min(max(0.01*coeff2,min2),max2);
    // bool cond2_a = (delta_new / delta_old) > thresh2; // if "new" diff than "old"
    // bool cond2_a = abs(delta_new - delta_old) > 0.05;
    // bool cond2_a = abs(delta_new - delta_old) > 0.01;
    // bool cond2_a = abs(delta_new - delta_old)/delta_old > 0.75;
    // bool cond2_a = (delta_pair/var_app_f > 0.0002);
    // [0. 4]
    // [0.125 2.57]
    // [90% @ 0.125; 50% @ 0.675; 20% @ 1.285; 10% @ 2.57]
    // bool cond2_a = (delta_pair/std_for_ztest > 0.125); // two-tailed z-test
    // bool cond2_a = (delta_pair/std_for_ztest > 0.01); // two-tailed z-test
    // bool cond2_a = (delta_pair/std_for_ztest > thresh2_z); // two-tailed z-test
    // bool cond2_a = (delta_pair/std_for_ztest > 1.285); // two-tailed z-test
    // bool cond2_a = (delta_pair/std_for_ztest > thresh2_z); // two-tailed z-test
    // bool cond2_a = (delta_pair/std_for_ztest > 0.675); // two-tailed z-test
    bool cond2_a = (delta_pair/std_for_ztest > 0.675); // two-tailed z-test
    // bool cond2_b = is_prop ? (delta_new > thresh2) : true;
    // if (cond2_a and cond2_b){
    if (cond2_a){
      sm_helper[k].lprob_f_app = -10;
      atomicAdd(&count_rules[2],1);
      return;
    }
    
    // -- [3] split if var too big --
    // float coeff3 = (size_ratio > 1) ? 1.0/size_ratio : 1.;
    // float coeff3 = (size_ratio > 1) ? 1.0/(5.*size_ratio) : 1.0;
    // float min3 = 10.0;
    // float max3 = 100.0;
    // float thresh3 = min(max(3.0*coeff3,min3),max3);
    // if ((var_app_f / 0.002) > thresh3){
    //   sm_helper[k].lprob_f_app = -10;
    //   atomicAdd(&count_rules[3],1);
    //   return;
    // }

    // -- [4] split if delta var is big --
    float thresh4 = max(0.5 - lsize_ratio,0.25);
    bool cond4_a = abs(var_app_f  - var_app_new)/var_app_f > thresh4;
    bool cond4_b = abs(var_app_f  - var_app_old)/var_app_f > thresh4;
    bool cond4_c = var_app_f > 0.002;
    if ((cond4_a and cond4_b) and cond4_c){
      sm_helper[k].lprob_f_app = -10;
      atomicAdd(&count_rules[4],1);
      return;
    }

    // -- no split --
    sm_helper[k].lprob_f_app = 10;
    atomicAdd(&count_rules[5],1);
    return;


    // // printf("var_app_[s,k,f]: %2.3lf %2.3lf %2.3lf\n",
    // //        var_app_s*100,var_app_k*100,var_app_f*100);

    // // -- prior --
    // double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // // double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // // double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // double3 _prior_icov = sp_params[k].sample_sigma_shape;
    // double3 prior_icov=add_sigma_smoothing(_prior_icov,prior_count,prior_count,sp_size);
    // // double3 _prior_icov = sp_params[k].prior_icov;
    // // double3 prior_icov=add_sigma_smoothing(_prior_icov,prior_count,prior_count,sp_size);
    // // double3 prior_icov = sp_params[k].prior_icov;
    // double3 sigma = prior_icov;
    // double nf = 50;
    // // double total_count = 1.*(prior_count + nf);

    // float pc = prior_count;
	// float pc_sqrt = sqrt(sp_params[k].prior_count);
    // double pr_det = prior_icov.x * prior_icov.z  - prior_icov.y * prior_icov.y;
    // pr_det = sqrt(pr_det);
    // float perc_invalid = sp_params[k].icount;
    // perc_invalid = min(max(perc_invalid,0.),1.0); // from invalidate_disc
    // // if (perc_invalid<0){
    // //   printf("INVALID[%d] %2.3f\n",k,perc_invalid);
    // // }
    // // assert(perc_invalid>=0);
    // // prior_icov.x = pc/pr_det*prior_icov.x;
    // // prior_icov.y = pc/pr_det*prior_icov.y;
    // // prior_icov.z = pc/pr_det*prior_icov.z;
    // // prior_icov.x = prior_icov.x;
    // // prior_icov.y = prior_icov.y;
    // // prior_icov.z = prior_icov.z;


	// double total_count = (double)prior_count*51;
    // // sigma_shape.x = (pc*sp_size + sigma_shape.x) / (total_count + 3.0);
    // // sigma_shape.y = (sigma_shape.y) / (total_count + 3.0);
    // // sigma_shape.z = (pc*sp_size + sigma_shape.z) / (total_count + 3.0);
    // // prior_sigma_shape.x = (pc*sp_size + prior_sigma_shape.x) / (total_count + 3.0);
    // // prior_sigma_shape.y = (prior_sigma_shape.y) / (total_count + 3.0);
    // // prior_sigma_shape.z = (pc*sp_size + prior_sigma_shape.z) / (total_count + 3.0);



    // // prior_sigma_shape.x = pc_sqrt/pr_det * prior_icov.x;
    // // prior_sigma_shape.y = pc_sqrt/pr_det * prior_icov.y;
    // // prior_sigma_shape.z = pc_sqrt/pr_det * prior_icov.z;
    // // prior_sigma_shape.x = (nf*sp_size + sigma.x)/(total_count + 3.0);
    // // prior_sigma_shape.y = (sigma.y) / (total_count + 3.0);
    // // prior_sigma_shape.z = (nf*sp_size + sigma.z)/(total_count + 3.0);
    // double3 _ucond_prior_sigma;
    // // ucond_prior_sigma.x = 1./sp_size;
    // // ucond_prior_sigma.y = 0;
    // // ucond_prior_sigma.z = 1./sp_size;
    // _ucond_prior_sigma.x = sp_size;
    // _ucond_prior_sigma.y = 0;
    // _ucond_prior_sigma.z = sp_size;
    // double3 ucond_prior_sigma=add_sigma_smoothing(_ucond_prior_sigma,prior_count,
    //                                               prior_count,sp_size);


    // // -- shape info --
    // double3 sigma_k = sm_helper[k].sigma_k;
    // double3 sigma_s = sm_helper[k].sigma_s;
    // double3 sigma_f = sm_helper[k].sigma_f;
    // // double3 sigma_k = compute_sigma_shape(sm_helper[k].sum_shape,
    // //                                       sm_helper[k].sq_sum_shape,
    // //                                       count_k,prior_count,sp_size);
    // // double3 sigma_s = compute_sigma_shape(sm_helper[s].sum_shape,
    // //                                       sm_helper[s].sq_sum_shape,
    // //                                       count_s,prior_count,sp_size);
    // // int2 sum_shape_f = get_sum_shape(sm_helper[s].sum_shape,sm_helper[k].sum_shape);
    // // longlong3 sq_sum_shape_f = get_sq_sum_shape(sm_helper[s].sq_sum_shape,
    // //                                             sm_helper[k].sq_sum_shape);
    // // double3 sigma_f = compute_sigma_shape(sum_shape_f,sq_sum_shape_f,
    // //                                       count_f,prior_count,sp_size);

    // // -- marginal likelihood --
    // // double lprob_k = marginal_likelihood_app_p(sum_k,sq_sum_k,count_k,sigma2_app);
    // // double lprob_s = marginal_likelihood_app_p(sum_s,sq_sum_s,count_s,sigma2_app);
    // // double lprob_f = marginal_likelihood_app_p(sum_f,sq_sum_f,count_f,sigma2_app);
    // // double sigma2_prior_var = 1.;
    // // double lprob_k = appearance_variance_p(sum_k,sq_sum_k,count_k,sigma2_prior_var);
    // // double lprob_s = appearance_variance_p(sum_s,sq_sum_s,count_s,sigma2_prior_var);
    // // double lprob_f = appearance_variance_p(sum_f,sq_sum_f,count_f,sigma2_prior_var);
    // // double sigma2_prior_var = 1.;
    // // double lprob_app_k = compute_lprob_mu_app(sum_k,mu_prior,
    // //                                           count_k,sigma2_prior_var);
    // // double lprob_app_s = compute_lprob_mu_app(sum_s,mu_prior,
    // //                                           count_s,sigma2_prior_var);
    // // double lprob_app_f = compute_lprob_mu_app(sum_f,mu_prior,
    // //                                           count_f,sigma2_prior_var);

    // // -- shape --
    // // double lprob_s_cond = marginal_likelihood_shape_p(sigma_s, prior_sigma_shape,
    // //                                                   prior_count,count_s);
    // // double lprob_s_ucond = marginal_likelihood_shape_p(sigma_s, ucond_prior_sigma,
    // //                                                    prior_count,count_s);
    // // double lprob_k_cond = marginal_likelihood_shape_p(sigma_k, prior_sigma_shape,
    // //                                                   prior_count,count_k);
    // // double lprob_k_ucond = marginal_likelihood_shape_p(sigma_k, ucond_prior_sigma,
    // //                                                    prior_count,count_k);
    // // double lprob_f = marginal_likelihood_shape_p(sigma_f, prior_sigma_shape,
    // //                                              prior_count,count_f);


    // // -- size dep. weight --
    // int sp2 = sp_size*sp_size;
    // // double size_x = count_f - 1.*sp2
    // double coeff = 0.005;
    // double w0 = 1.0/(1 + exp(-coeff*(sp2 - count_f)))*1.;
    // w0 = (w0+0.5)/1.5;
    // double coeff_00 = 0.05;
    // double w_00 = 1.0/(1 + exp(-coeff_00*(sp2 - count_f)))*1.;
    // // double w1 = 1.0/(1 + exp(-coeff*(sp2 - count_f)))*1.;
    // double w1 = 1-w0;
    // // double w = min(max(count_f/(1.0*sp2),0.01),10000.0)/10.0;
    // // double w = min(max(count_f/(1.0*sp2),0.01),10000.0)/1.0*2.0;
    // double w;
    // double size_weight = sp2*10.;
    // if (count_f < sp2){ w = 1.0; }
    // else{
    //   // w = exp((count_f - sp2)/size_weight);
    //   w = exp(count_f/(5.*sp2))/10.;
    // }
    // // w = w*w0;
    // w = 1.0;
    // double w3 = 1.0;
    // double clipped_w1 = min(max(w1,0.001),1.0);
    // double clipped_w0 = min(max(w0,0.25),1.0);

    // // -- shape prior --
    // double lprob_f = w1*compute_lprob_sigma_shape(sigma_f,prior_icov) - 1e-6;
    // double lprob_s_cond = w1*compute_lprob_sigma_shape(sigma_s,prior_icov);
    // double lprob_s_ucond = w1*compute_lprob_sigma_shape(sigma_s,ucond_prior_sigma);
    // double lprob_k_cond = w1*compute_lprob_sigma_shape(sigma_k,prior_icov);
    // double lprob_k_ucond = w1*compute_lprob_sigma_shape(sigma_k,ucond_prior_sigma);
    // // lprob_f = -0.8; // a fixed offset for flexibility; allow "change by 20%"
    // // lprob_f = -1.0; // a fixed offset for flexibility; allow "change by 20%"
    // // lprob_f = -0.4;
    // // lprob_f = -0.0;

    // // if ((k > 100) and (k < 110)){
    // double2 mu_shape = sp_params[k].mu_shape;
    // double mu_s_x = mu_shape.x;
    // double mu_s_y = mu_shape.y;
    // // bool bounds = (abs(mu_s_x - 810)<20) and (abs(mu_s_y - 105) < 20);
    // // bool bounds = (abs(mu_s_x - 670)<20) and (abs(mu_s_y - 90) < 20);
    // // bool bounds = (abs(mu_s_x - 230)<60) and (abs(mu_s_y - 418) < 20);
    // // bool bounds = (abs(mu_s_x - 850)<30) and (abs(mu_s_y - 196) < 30);
    // // bool bounds = (abs(mu_s_x - 847)<12) and (abs(mu_s_y - 100) < 90);
    // // bool bounds = (abs(mu_s_x - (854-50))<100) and (abs(mu_s_y - 0) < 30);
    // // bool bounds = (abs(mu_s_x - (854-0))<100) and (abs(mu_s_y - 0) < 30);
    // // if (count_f > 10000){
    // // if (bounds){
    // if (false){
        
    //   // printf("[bounds-tag:%d] %2.3lf | %2.3lf %2.3lf | %2.3lf %2.3lf\n",
    //   //        k,lprob_f,lprob_s_cond,lprob_k_cond,lprob_s_ucond,lprob_k_ucond);
    //   printf("[bounds-tag:%d@(%d,%d:%2.2lf,%2.2lf)] %2.3lf %2.3lf %2.3lf|%2.3lf %2.3lf %2.3lf|%2.3lf %2.3lf %2.3lf|%2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | [%2.3lf | %2.3lf %2.3lf | %2.3lf %2.3lf]\n",
    //          k,count_f,int(prior_count),mu_s_x,mu_s_y,
    //          sigma_f.x,sigma_f.y,sigma_f.z,
    //          sigma_s.x,sigma_s.y,sigma_s.z,
    //          sigma_k.x,sigma_k.y,sigma_k.z,
    //          prior_icov.x,prior_icov.y,prior_icov.z,
    //          prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,
    //          ucond_prior_sigma.x,ucond_prior_sigma.y,ucond_prior_sigma.z,
    //          lprob_f,lprob_s_cond,lprob_k_cond,lprob_s_ucond,lprob_k_ucond);
    //          // ucond_prior_sigma.x,ucond_prior_sigma.y,ucond_prior_sigma.z);
    // }

    // // -- prob --
    // // double lprob_s_cond = 0;
    // // double lprob_s_ucond = 0;
    // // double lprob_k_cond = 0;
    // // double lprob_k_ucond = 0;
    // // double lprob_f = 0;

    // // lprob_s_cond = 0;
    // // lprob_s_ucond = 0;
    // // lprob_k_cond = 0;
    // // lprob_k_ucond = 0;
    // // lprob_f = 0;

    // // -- variance --
    // // double var_scale = 10000*1.6;//w/10.;
    // // double var_scale = 10000*1.7;//w/10.;
    // // double var_scale = 10000*1.6;//w/10.; // so big b/c the gaussian prior on size
    // // double var_scale = w*16.;//w/10.;
    // // double var_scale = w*16.;//w/10.;
    // // double var_scale = w*16.5;//w/10.;
    // double var_scale = 1.;
    // // lprob_s_cond_app += var_scale * -var_app_s;
    // // lprob_k_cond_app += var_scale * -var_app_k;
    // // lprob_f_app += var_scale * -var_app_f;

    // double sigma_var_prior = 1.0/w3;
    // double var_prior_f = -(var_app_f - 0.004)*(var_app_f - 0.004)/sigma_var_prior;
    // double var_prior_s = -(var_app_s - 0.004)*(var_app_s - 0.004)/sigma_var_prior;
    // double var_prior_k = -(var_app_k - 0.004)*(var_app_k - 0.004)/sigma_var_prior;

    // // lprob_f += var_prior_f;//*sqrt(1.0*count_f);
    // // lprob_s_cond += var_prior_s;//*sqrt(1.0*count_s);
    // // lprob_s_ucond += var_prior_s;//*sqrt(1.0*count_s);;
    // // lprob_k_cond += var_prior_k;//*sqrt(1.0*count_k);
    // // lprob_k_ucond += var_prior_k;//*sqrt(1.0*count_k);

    // // double c = 0.01;
    // // double c = 0.05; // stable on 4 of 6
    // // double c = 0.1;
    // // double c = 0.20;
    // // float c = min(max(log(w)/2.,0.25),1.0);
    // // double c = 0.20;
    // // double c = 2.0;
    // // double c = 1.5;
    // // double c = 1.0;
    // // double c = 5.0;
    // // double c = 8.0;
    // // double c = 1.0*same_mean;
    // double c = 2.0*same_mean;
    // // double c = 10.0*same_mean;
    // // double c = 0.5;
    // // double c = 2.0*clipped_w0;
    // // double c = 0.5;//*clipped_w0;
    // // double delta_var_f = 2*(exp(c*sqrt(var_app_f)/sqrt(0.004))-exp(c))/exp(c*sqrt(0.01)/sqrt(0.004));
    // // double delta_var_f = 2*exp(c*var_app_f/sqrt(0.004));

    // double delta_var_f = 2*(exp(c*var_app_f/0.004)-exp(0.0))/(exp(c)-exp(0.0));
    // double var_delta = (abs(var_app_f - var_app_s) + abs(var_app_f - var_app_k))/(var_app_f*clipped_w1);
    // // lprob_f += -var_delta/10.;
    // // float iterm_b = 0.25;//max(log(w),1.);
    // // float iterm_b = 0.25;//max(log(w),1.);
    // // float iterm_b = min(max(log(w)/2.,0.25),1.0);
    // // float iterm_b = min(max(log(w),0.1),3.0);
    // // float iterm_b = min(max(log(w)/2.,0.25),1.0);
    // // float iterm_b = 0.25;//max(log(w),1.);
    // // float iterm_b = 0.5;//max(log(w),1.);
    // float iterm_b = w;

    // // -- percent invalid means less if spix is small --
    // //perc_invalid =count_f<(sp_size*sp_size)?perc_invalid/4. : perc_invalid;
    // // perc_invalid = count_f<(sp_size*sp_size)?perc_invalid/4. : perc_invalid;
    // // perc_invalid = w0*perc_invalid;
    // // perc_invalid = w_00*perc_invalid;

    // // -- percent invalid weight --
    // float iterm = exp((perc_invalid-1.0)/(iterm_b))-exp(-1.0/(iterm_b));
    // iterm = iterm / (exp(0.0) - exp(-1.0/iterm_b))+0.01;
    // // float iterm = exp((1.0-1.0)/(iterm_b))-exp(-1.0/(iterm_b));
    // double var_term = var_delta*iterm*delta_var_f*w;
    // var_term = (var_app_f > 1.0) ? 100 : var_term; // handle absurd cases
    // lprob_f += -var_term;
    // // double var_delta = var_scale*(abs(var_app_f - var_app_s) + abs(var_app_f - var_app_k))/var_app_f;
    // // var_delta = 10.*max((var_delta - 1.1),0.);
    // double b = 2.0;
    // // double b = 1*(max(log(w),-1.0)+1);
    // // var_delta = (w/10.)*(max(exp(b*(var_delta-1.))-1.2,0.0));

    // // -- greater than term --
    // // lprob_f += (var_app_s > (var_app_f*1.5)) ? -100 : 0;
    // // lprob_f += (var_app_k > (var_app_f*1.5)) ? -100 : 0;
    // // lprob_s_ucond += (var_app_s > var_app_f) ? -100 : 0;
    // // lprob_k_ucond += (var_app_k > var_app_f) ? -100 : 0;

    // // -- append appearance to cond --
    // double sigma2_prior_var = 1.;
    // // double w=1.2*min(max(sqrt(1.*count_f)/(1.*sp_size),10.0),100.0);///(1.*var_app_f);
    // // double w = 1.0;
    // double sum_app = (abs(mu_prior.x)+abs(mu_prior.y)+abs(mu_prior.z))/3.0;
    // // double clipped_w = 1.0;//min(max(w1,0.001),1.0);
    // // sigma2_prior_var = 1.0/(0.009*clipped_w1);
    // // sigma2_prior_var = 1.0/(0.009*clipped_w1*10.);
    // // sigma2_prior_var = 1.0/(0.009*clipped_w1*10.);
    // // sigma2_prior_var = 0.009/sqrt(1.0*count_f)*100.;
    // // sigma2_prior_var = var_app_f/sqrt(1.0*count_f);
    // sigma2_prior_var = 0.009;
    // double lprob_f_app = w*compute_l2norm_mu_app_p(sum_f,mu_prior,
    //                                                count_f,sigma2_prior_var);
    // // lprob_f_app = lprob_f_app - 1e-6;
    // double lprob_s_cond_app = w*compute_l2norm_mu_app_p(sum_s,mu_prior,count_s,
    //                                                      sigma2_prior_var);
    // // lprob_s_cond_app = max(lprob_s_cond_app,-100.);
    // double lprob_k_cond_app = w*compute_l2norm_mu_app_p(sum_k,mu_prior,count_k,
    //                                                      sigma2_prior_var);
    // // lprob_k_cond_app = max(lprob_k_cond_app,-100.);
    // // lprob_f_app = -1.0;
    // // lprob_f_app = -lprob_f_app;
    // // lprob_f_app = -0.0;

    // // lprob_f_app = 0.0;
    // // lprob_s_cond_app = 0.0;
    // // lprob_k_cond_app = 0.0;

    // // if ((k > 100) and (k < 110)){
    // // bool bounds2 = (abs(mu_s_x - 695)<20) and (abs(mu_s_y - 60) < 20);
    // // bool bounds2 = (abs(mu_s_x - 671)<20) and (abs(mu_s_y - 416) < 20);
    // // bool bounds2 = (abs(mu_s_x - 230)<60) and (abs(mu_s_y - 418) < 20);
    // // bool bounds2 = (abs(mu_s_x - 847)<12) and (abs(mu_s_y - 100) < 90);
    // // bool bounds2 = (abs(mu_s_x - (854-50))<100) and (abs(mu_s_y - 0) < 30);
    // bool bounds2 = (abs(mu_s_x - (854-0))<100) and (abs(mu_s_y - 0) < 30);
    // if (count_f > 1000){
    // // if (bounds2){
    //   printf("[k-tag:%d]: %2.3lf %d %2.3lf %2.2f %2.4f %2.4f %2.4lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf (%2.3lf %2.3lf) (%2.3lf %2.3lf) \n",
    //          k,
    //          w,count_f,var_delta,perc_invalid,iterm,iterm_b,delta_var_f,
    //          lprob_f_app,lprob_s_cond_app,lprob_k_cond_app,
    //          var_app_f,var_app_s,var_app_k,
    //          lprob_f,lprob_s_cond,lprob_s_ucond,lprob_k_cond,lprob_k_ucond);
    // }
    // // lprob_f_app += -var_delta; // a "big" negative term is subtracted => split!

    // // lprob_f_app += var_scale * (var_app_s + var_app_k - var_app_f)/var_app_f;
    // // lprob_s_cond += compute_l2norm_mu_app_p(sum_s,mu_prior,
    // //                                         count_s,sigma2_prior_var);
    // // lprob_k_cond += compute_l2norm_mu_app_p(sum_k,mu_prior,
    // //                                         count_k,sigma2_prior_var);
    // // lprob_f += compute_l2norm_mu_app_p(sum_f,mu_prior,
    // //                                    count_f,sigma2_prior_var);

    // // -- size propr --
    // // float _sp_size = 1.*sp_size;
    // float pr_c2 = sqrt(1.*prior_count/2.);
    // float _sp_size_v0 = prop ? sqrt(1.*prior_count) : sp_size;
    // // float _sp_size_v1 = prop ? pr_c2 : 1.* sp_size;
    // // float _sp_size_v1 = prop ? pr_c2 : 1.* sp_size;
    // float _sp_size_v2 = prop ? pr_c2 : 1.* sp_size;
    // // float _sp_size_v1 = 1.* pr_c2;
    // // float _sp_size_v2 = 1.* sp_size;
    // float _sp_size_v1 = prop ? 1.* sqrt(1.*prior_count) : sp_size;
    // // float _sp_size_v2 = 1.* sp_size;
    // // float _sp_size_v2 = 1.* sp_size;
    // // float _sp_size_v2 = prop ? pr_c2 : sp_size;
    // // float _sp_size_v1 = 1.*sp_size;
    // // float _sp_size_v1 = 1.*prior_count;

    // // _sp_size_v0 = sp_size;
    // // _sp_size_v1 = sp_size;
    // // _sp_size_v2 = sp_size;
    // float normz = prior_count;
    // // lprob_f += size_likelihood_p(count_f,_sp_size_v0,sigma2_size)/normz;
    // // lprob_s_cond += size_likelihood_p(count_s,_sp_size_v1,sigma2_size)/normz;
    // // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size_v2,sigma2_size);
    // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size_v2,sigma2_size)/normz;
    // // lprob_k_cond += size_likelihood_p(count_k,_sp_size_v1,sigma2_size)/normz;
    // // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size_v2,sigma2_size);
    // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size_v2,sigma2_size)/normz;

    // // -- gaussian prior --
    // float _sp_size = 2.*sp_size; // set me to "1" for great results.
    // // float sigma2_gauss = 0.01;
    // float sigma2_gauss = 1.;
    // normz = sqrt(_sp_size);
    // // lprob_f += size_likelihood_p(count_f,_sp_size,sigma2_gauss)/normz;
    // // lprob_s_cond += size_likelihood_p(count_s,_sp_size,sigma2_gauss)/normz;
    // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size,sigma2_gauss)/normz;
    // // lprob_k_cond += size_likelihood_p(count_k,_sp_size,sigma2_gauss)/normz;
    // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size,sigma2_gauss)/normz;

    // // -- fixed prior too ? --
    // // float init_size = 1.*sp_size;
    // // lprob_f += size_likelihood_p(count_f,_sp_size_v0,sigma2_size)/normz;
    // // lprob_s_cond += size_likelihood_p(count_s,_sp_size_v1,sigma2_size)/normz;
    // // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size_v2,sigma2_size);
    // // lprob_s_ucond += size_likelihood_p(count_s,_sp_size_v2,sigma2_size)/normz;
    // // lprob_k_cond += size_likelihood_p(count_k,_sp_size_v1,sigma2_size)/normz;
    // // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size_v2,sigma2_size);
    // // lprob_k_ucond += size_likelihood_p(count_k,_sp_size_v2,sigma2_size)/normz;

    // // -- include size term --
    // // lprob_k += size_beta_likelihood_p(count_k,sp_size,sigma2_size,npix);
    // // lprob_s += size_beta_likelihood_p(count_s,sp_size,sigma2_size,npix);
    // // lprob_f += size_beta_likelihood_p(count_f,sp_size,sigma2_size,npix);


    // // -- offset by size and variance --
    // w = 1.0 - min(count_f / (2.0*sp_size*sp_size),1.0);
    // // double var_offset = var_app_f / (10.0*0.004);
    // double var_offset = (var_app_f < 0.05) ? 1.0 : 0;
    // // w = w + 10.*var_offset;

    // // -- [shape] write --
    // sm_helper[k].lprob_f_shape = lprob_f + w;
    // sm_helper[k].lprob_s_cond_shape = lprob_s_cond;
    // sm_helper[k].lprob_s_ucond_shape = lprob_s_ucond;
    // sm_helper[k].lprob_k_cond_shape = lprob_k_cond;
    // sm_helper[k].lprob_k_ucond_shape = lprob_k_ucond;

    // // -- [app] write --
    // sm_helper[k].lprob_s_cond_ap p= lprob_s_cond_app;
    // sm_helper[k].lprob_k_cond_app = lprob_k_cond_app;
    // sm_helper[k].lprob_f_app = lprob_f_app;
    // // sm_helper[k].lprob_s_cond_app = 0;
    // // sm_helper[k].lprob_k_cond_app = 0;
    // // sm_helper[k].lprob_f_app = 0;

    // // if (count_f > 1312){
    // //   printf("[%d]: %d %d %d | %2.3lf %2.3lf %2.3lf\n",
    // //          k,count_f,count_s,count_k,lprob_f,lprob_s_cond,lprob_k_cond);
    // // }
    // // printf("[%d]: %lf %lf %lf | %lf %lf | %lf %lf %lf | %lf %lf %lf\n",
    // //        k,lprob_f,lprob_s_cond,lprob_k_cond,
    // //        lprob_s_ucond,lprob_k_ucond,
    // //        sigma_f.x,sigma_f.y,sigma_f.z,
    // //        prior_icov.x,prior_icov.y,prior_icov.z);

}

__device__ double size_likelihood_p(int curr_count, float tgt_count, double sigma2) {
  // double delta = 1.*(curr_count - tgt_count*tgt_count)/100.;
  double delta = 1.*(sqrt(1.*curr_count) - tgt_count);
  // double lprob = - log(2*M_PI*sigma2)/2. - delta*delta/(2*sigma2);
  // double lprob = - sqrt(delta*delta);
  double delta2 = delta*delta;
  double lprob = -delta*delta/sigma2;
  // if (delta2 > 100){
  //   printf("[size_likelihood_p] %d, %2.2f, %2.2f, %2.3lf\n",curr_count,sqrt(1.*curr_count),tgt_count,delta2);
  // }

  // return 0.;
  return lprob;
  // printf("size is 0!\n");


}

__device__ double size_likelihood_p_b(int curr_count, float tgt_count, double sigma2) {
  // double delta = 1.*(curr_count - tgt_count*tgt_count)/100.;
  double delta = 1.*(sqrt(1.*curr_count) - tgt_count);
  // delta = (delta > 0) ? delta : 0;
  delta = (delta > (-tgt_count)) ? delta : 0;
  // double lprob = - log(2*M_PI*sigma2)/2. - delta*delta/(2*sigma2);
  // double lprob = - sqrt(delta*delta);
  // double delta2 = delta*delta;
  double lprob = -delta*delta/sigma2;
  // if (delta2 > 100){
  //   printf("[size_likelihood_p] %d, %2.2f, %2.2f, %2.3lf\n",curr_count,sqrt(1.*curr_count),tgt_count,delta2);
  // }

  // return 0.;
  return lprob;
  // printf("size is 0!\n");


}

__device__ double size_beta_likelihood_p(int _count, int _tgt_count,
                                       double alpha, const int _npix) {
  if (alpha < 0){ return 0; }
  double count = 1.*_count;
  double npix = 1.*_npix;
  double tgt_count = 1*_tgt_count*_tgt_count;
  double beta = alpha*(npix-tgt_count)/(tgt_count+1e-10); // just in case...
  // double beta = alpha;
  double lprob = (alpha-1)*log(count/npix) + (beta-1)*log(1-count/npix);
  // lprob += lgammaf(npix*alpha/tgt_count) - lgammaf(alpha) - lgammaf(beta);
  lprob += lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta);
  return lprob;
}


__device__ double compute_lprob_sigma_shape(double3 sigma_est,
                                            double3 prior_sigma) {

  double lprob = -wasserstein_p(sigma_est,prior_sigma);
  // double lprob = 0;
  // lprob = 0;
  return lprob;
}

__device__ double marginal_likelihood_shape_p(double3 sigma_est, double3 prior_sigma,
                                              float pr_count,int num_obs) {

  // -- compute det --
  double sxx = sigma_est.x;
  double sxy = sigma_est.y;
  double syy = sigma_est.z;
  double det_est = abs(sxx*syy-sxy*sxy);
  assert(det_est>0.001);

  double pr_sxx = prior_sigma.x;
  double pr_sxy = prior_sigma.y;
  double pr_syy = prior_sigma.z;
  double det_pr = abs(pr_sxx*pr_syy-pr_sxy*pr_sxy);
  assert(det_pr>0.001);

  // -- compute marginal --
  if (det_est < 0.001){ return -10000000; }
  double post_count = 1.*pr_count + num_obs;
  double gamma2_post = lgamma(post_count/2) + lgamma((post_count-1)/2);
  double gamma2_pr = lgamma(pr_count/2) + lgamma((pr_count-1)/2);
  double h_const = num_obs * log(M_PI) + log(pr_count) - log(post_count);
  // double lprob = pr_count*log(det_pr) - post_count*log(det_post) + gamma2_post - gamma2_pr + h_const;
  double lprob = pr_count*log(det_pr) - post_count*log(det_est);
  return lprob;
}


__device__ double wasserstein_p(double3 sigma_est,double3 sigma_prior) {

  // -- ... --
  // printf("s11,s12,s22 | %lf %lf %lf | %lf %lf %lf \n",
  //        sigma_est.x,sigma_est.y,sigma_est.z,
  //        sigma_prior.x,sigma_prior.y,sigma_prior.z);

  // Step 1: Compute eigenvalues for sigma_est
  double3 eigen_est = eigenvals_cov_p(sigma_est);
  double lambda1_est = eigen_est.x;
  double lambda2_est = eigen_est.y;
  double detA = lambda1_est * lambda2_est;

  // Step 2: Compute eigenvalues for sigma_prior
  double3 eigen_prior = eigenvals_cov_p(sigma_prior);
  double lambda1_prior = eigen_prior.x;
  double lambda2_prior = eigen_prior.y;
  double detB = lambda1_prior * lambda2_prior;

  // Step 3: Computer eigenvalues for C
  double3 eigen_cross = eigenvals_cov_pair(sigma_est, sigma_prior, detA, detB);
  double trace_cross = eigen_cross.z;

  // Step 4: Wasserstein squared distance
  double wasserstein_distance_squared = eigen_est.z + eigen_prior.z - 2*trace_cross;

  // Return the square root to get the actual Wasserstein distance
  return wasserstein_distance_squared;

}

__device__ double3 eigenvals_cov_pair(double3 icovA, double3 icovB,
                                      double detA, double detB){
  // -- get det and trace --
  double determinant = detA*detB;
  double trace = (icovA.x * icovB.x) + 2*(icovA.y * icovB.y) + (icovA.z * icovB.z);

  // Calculate the square root term
  double tmp = (trace * trace)/4.0;
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

  // -- compute sqrt --
  lambda1 = (lambda1 > 0) ? sqrt(lambda1) : 0;
  lambda2 = (lambda2 > 0) ? sqrt(lambda2) : 0;
  trace = lambda1 + lambda2;

  return make_double3(lambda1, lambda2, trace);
}

__device__ double3 eigenvals_cov_p(double3 icov) {

  // -- unpack --
  double s11 = icov.x;
  double s12 = icov.y;
  double s22 = icov.z;

  // Calculate the trace and determinant
  // double determinant = 1./(s11 * s22 - s12 * s12); // inverse cov rather than cov
  double determinant = (s11 * s22 - s12 * s12); // inverse cov rather than cov
  // double trace = (s11 + s22)*determinant; // "divide each term by det.
  double trace = (s11 + s22);
  //printf("s11,s12,s22,det,trace: %lf %lf %lf %lf %lf\n",s11,s12,s22,determinant,trace);

  // // -- info --
  // printf("sxx,sxy,syy,pc: %lf %lf %lf %f %lf | %d %d | %lld %lld %lld\n",
  //        sigma.x,sigma.y,sigma.z,prior_count,count,
  //        sum.x,sum.y,sq_sum.x,sq_sum.y,sq_sum.z);
  // assert(determinant>0.0001);
  // printf("s11,s22,det,trace: %lf %lf %lf %lf\n",s11,s22,determinant,trace);

  // Calculate the square root term
  double tmp = (trace * trace)/4.0;
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


__device__ double compare_mu_pair(double3 mu0,double3 mu1){
  double delta_x = (mu0.x - mu1.x);
  double delta_y = (mu0.y - mu1.y);
  double delta_z = (mu0.z - mu1.z);
  double l2norm = (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)/3.;
  return l2norm;
}

__device__ double compare_mu_app_pair(double3 mu0,double3 mu1, int count0, int count1){
  double delta_x = (mu0.x/count0 - mu1.x/count1);
  double delta_y = (mu0.y/count0 - mu1.y/count1);
  double delta_z = (mu0.z/count0 - mu1.z/count1);
  double l2norm = (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)/3.;
  return l2norm;
}

__device__ double compute_l2norm_mu_app_p(double3 sum_obs,float3 prior_mu,
                                          int _num_obs, double sigma2) {
  double num_obs = 1.*_num_obs;
  double delta_x = (sum_obs.x/num_obs - prior_mu.x);
  double delta_y = (sum_obs.y/num_obs - prior_mu.y);
  double delta_z = (sum_obs.z/num_obs - prior_mu.z);
  // if (abs(delta_x) < 0.01){
  //   printf("l2norm: %2.4lf %2.4lf\n",sum_obs.x/num_obs,prior_mu.x);
  //   assert(1==0);
  // }
  double l2norm = (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)/3.;
  return -3*7*l2norm/sigma2;
  // return 0.;
}


// __device__ double compute_lprob_mu_app(double3 sum_obs,float3 prior_mu,
//                                       int _num_obs, double sigma2) {
//   double num_obs = 1.*_num_obs;
//   double delta_x = (sum_obs.x/num_obs - prior_mu.x);
//   double delta_y = (sum_obs.y/num_obs - prior_mu.y);
//   double delta_z = (sum_obs.z/num_obs - prior_mu.z);
//   double lprob = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z;
//   lprob = -lprob/(2*sigma2);
//   return lprob;

// }

__device__ double appearance_variance_p(double3 sum_obs,double3 sq_sum_obs,
                                      int _num_obs, double sigma2) {
  double num_obs = 1.*_num_obs;
  double sample_var = (sq_sum_obs.x  - sum_obs.x*sum_obs.x);
  sample_var += (sq_sum_obs.y  - sum_obs.y*sum_obs.y);
  sample_var += (sq_sum_obs.z  - sum_obs.z*sum_obs.z);
  sample_var = sample_var/(3.*num_obs); // estimate sigma2
  // sample_var = sample_var/3.; // estimate sigma2
  // double lprob = -sample_var/sigma2;
  double lprob = 0.; // dev
  return lprob;
}

__device__ double marginal_likelihood_app_p(double3 sum_obs,double3 sq_sum_obs,
                                          int _num_obs, double sigma2) {
  // ref: from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
  // Equation 55 with modifications from Eq 57 where kappa = 1
  // -- silly; we should just replace forumla with tau2 -> infty limit --
  double tau2 = 1000.; // ~= mean has 95% prob to be within (-1,1)
  float num_obs = (float)_num_obs;

  // float3 mu_prior;

  double lprob_num = 1/2. * log(sigma2) - num_obs/2.0 * log(2*M_PI*sigma2) \
    - log(num_obs * tau2 + sigma2)/2.;
  double denom = 2*(num_obs*tau2+sigma2);
  double3 lprob;
  lprob.x = lprob_num - sq_sum_obs.x/(2*sigma2) \
    + tau2*sum_obs.x*sum_obs.x/(sigma2*denom);
  lprob.y = lprob_num - sq_sum_obs.y/(2*sigma2)
    + tau2*sum_obs.y*sum_obs.y/(sigma2*denom);
  lprob.z = lprob_num - sq_sum_obs.z/(2*sigma2)
    + tau2*sum_obs.z*sum_obs.z/(sigma2*denom);

  double _lprob;
  _lprob = lprob.x+lprob.y+lprob.z;
  return _lprob;
}




__global__  void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
                                        spix_params* sp_params,
                                        const int npix, const int nbatch,
                                        const int width, const int height,
                                        const int direction){
  // todo: add nbatch
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    if(!border[idx]) return;
    int x = idx % width;
    int y = idx / width;

    int C = seg[idx]; // center 
    int W; // north, south, east,west            
    W = OUT_OF_BOUNDS_LABEL; // init 

    if(direction==1){
      if ((y>1) && (y< height-2))
        {
          W = __ldg(&seg[idx+width]);  // down
        }
    }else{
      if ((x>1) && (x< width-2))
        {
          W = __ldg(&seg[idx-1]);  // left
        }
    }
        
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    bool prop_C = sp_params[C].prop;
    bool prop_W = (W>=0) ? sp_params[W].prop : false;
    // if (W>=0 && C!=W && (prop_C != prop_W)){
    if (W>=0 && C!=W && not(prop_C and prop_W)){
      atomicMax(&sm_pairs[2*C+1],W);
    }

    return;        
}

__global__
void calc_split_candidate_p(int* dists, int* spix, bool* border,
                          int distance, int* mutex, const int npix,
                          const int nbatch, const int width, const int height){
  
    // todo: add batch -- no nftrs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int x = idx % width;
    int y = idx / width;
    int C = dists[idx]; // center 
    int spixC = spix[idx];
    // if (border[idx]) return; 

    if(C!=distance) return;

    if ((y>0)&&(idx-width>=0)){
      if((dists[idx-width]==-1) and (spix[idx-width] == spixC)){
        dists[idx-width] = distance+1;
        mutex[0] = 1;
      }
    }          
    if ((x>0)&&(idx-1>=0)){
      if((dists[idx-1]==-1) and (spix[idx-1] == spixC)){
        dists[idx-1] = distance+1;
        mutex[0] = 1;
      }
    }
    if ((y<height-1)&&(idx+width<npix)){
      if((dists[idx+width]==-1) and (spix[idx+width] == spixC)){
        dists[idx+width] = distance+1;
        mutex[0] = 1;
      }
    }   
    if ((x<width-1)&&(idx+1<npix)){
      if((dists[idx+1]==-1) and (spix[idx+1] == spixC)){
        dists[idx+1] = distance+1;
        mutex[0] = 1;
      }
    }
    
    return;        
}


__global__ void init_split_p(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm_v2* sm_helper,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int direction,
                           const int* seg, int* max_sp, int max_spix) {

    // todo: add batch -- no nftrs
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    // *max_sp = max_spix+1;
    *max_sp = max_spix; // MAX number -> MAX label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int x;
    int y;
    if((direction==1)||(direction==-1))
    {
        x = int(sp_params[k].mu_shape.x)+direction;
        y = int(sp_params[k].mu_shape.y);
    }
    else
    {
        x = int(sp_params[k].mu_shape.x);
        y = int(sp_params[k].mu_shape.y)+direction;
    }
    
    int ind = y*width+x;
    if((ind<0)||(ind>width*height-1)) return;
    
    // if(border[ind]) return;
    if (seg[ind]!=k) return;
    seg_gpu[ind] = 1;

}


__global__ void calc_seg_split_p(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_spix) {
  // todo -- nbatch
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;
    int seg_val = __ldg(&seg[t]);

    if(sm_seg1[t]>__ldg(&sm_seg2[t])) seg_val += (max_spix+1); 
    sm_seg1[t] = seg_val;

    return;
}

__global__ void sum_by_label_merge_p(const float* img, const int* seg_gpu,
                                   spix_params* sp_params,
                                   spix_helper_sm_v2* sm_helper,
                                   const int npix, const int nbatch,
                                   const int width, const int nftrs) {
  // todo: nbatch
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;

	//get the label
	int k = __ldg(&seg_gpu[t]);
    float l = __ldg(& img[3*t]);
    float a = __ldg(& img[3*t+1]);
    float b = __ldg(& img[3*t+2]);
	//atomicAdd(&sp_params[k].count, 1); //TODO: Time it
	atomicAdd(&sm_helper[k].count, 1); 
	atomicAdd(&sm_helper[k].sq_sum_app.x, l*l);
	atomicAdd(&sm_helper[k].sq_sum_app.y, a*a);
	atomicAdd(&sm_helper[k].sq_sum_app.z,b*b);
    atomicAdd(&sm_helper[k].sum_app.x, l);
	atomicAdd(&sm_helper[k].sum_app.y, a);
	atomicAdd(&sm_helper[k].sum_app.z, b);
    
	int x = t % width;
	int y = t / width; 
	atomicAdd((unsigned long long *)&sm_helper[k].sum_shape.x, x);
	atomicAdd((unsigned long long *)&sm_helper[k].sum_shape.y, y);
    // atomicAdd(&sm_helper[k].sum_shape.x, x);
    // atomicAdd(&sm_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.x, x*x);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.y, x*y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.z, y*y);


}

__global__ void sum_by_label_split_p(const float* img, const int* seg,
                                     int* shifted, spix_params* sp_params,
                                     spix_helper_sm_v2* sm_helper,
                                     const int npix, const int nbatch,
                                     const int height, const int width,
                                     const int nftrs, int max_spix) {
  // todo: nbatch
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=npix) return;

	//get the label
    
	int k = __ldg(&seg[t]);
    float l = __ldg(& img[3*t]);
    float a = __ldg(& img[3*t+1]);
    float b = __ldg(& img[3*t+2]);
	atomicAdd(&sm_helper[k].count, 1); 
    atomicAdd(&sm_helper[k].sum_app.x, l);
	atomicAdd(&sm_helper[k].sum_app.y, a);
	atomicAdd(&sm_helper[k].sum_app.z, b);
	atomicAdd(&sm_helper[k].sq_sum_app.x, l*l);
	atomicAdd(&sm_helper[k].sq_sum_app.y, a*a);
	atomicAdd(&sm_helper[k].sq_sum_app.z,b*b);

    int shifted_k = __ldg(&shifted[t]);
    atomicAdd(&sm_helper[k].ninvalid,shifted_k<0);

    
	int x = t % width;
	int y = t / width; 
	atomicAdd((unsigned long long *)&sm_helper[k].sum_shape.x, x);
	atomicAdd((unsigned long long *)&sm_helper[k].sum_shape.y, y);
    // atomicAdd(&sm_helper[k].sum_shape.x, x);
    // atomicAdd(&sm_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.x, x*x);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.y, x*y);
    atomicAdd((unsigned long long *)&sm_helper[k].sq_sum_shape.z, y*y);
    return;
}





__global__
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                            spix_params* sp_params,
                            spix_helper* sp_helper,
                            spix_helper_sm_v2* sm_helper,
                            const int npix, const int nbatch,
                            const int width, const int nftrs,
                            const int nspix_buffer,
                            int sp_size, float log_alpha,
                            int max_spix, int* max_sp) {
  // todo -- add nbatch and nftrs
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
	// if (sp_params[k].prop == false) return;
    
    int s = k + (max_spix+1);
    if(s>=nspix_buffer) return;
    // float count_f = __ldg(&sp_params[k].count);
    float _count_f = __ldg(&sp_params[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_f = count_k + count_s;

    if((count_f<1)||(count_k<1)||(count_s<1)) return;

    // float lprob_k = __ldg(&sm_helper[k].numerator_app);
    // float lprob_s = __ldg(&sm_helper[s].numerator_app);
    // float lprob_f = __ldg(&sm_helper[k].numerator_f_app);

    // sm_helper[k].numerator_f_app = lprob_f;
    // sm_helper[k].denominator.x = lprob_s_cond;
    // sm_helper[k].denominator.y = lprob_s_ucond;
    // sm_helper[k].denominator_f.x = lprob_k_cond;
    // sm_helper[k].denominator_f.y = lprob_k_ucond;
    float prior_count_f = sp_params[k].prior_count; // dev only
    bool prop_k = sp_params[k].prop; // dev only

    // -- dev only --
    double2 mu_shape = sp_params[k].mu_shape;
    double mu_s_x = mu_shape.x;
    double mu_s_y = mu_shape.y;

    // -- unpack --
    float lprob_f = __ldg(&sm_helper[k].lprob_f_shape);
    float lprob_k_cond = __ldg(&sm_helper[k].lprob_k_cond_shape);
    float lprob_k_ucond = __ldg(&sm_helper[k].lprob_k_ucond_shape);
    float lprob_s_cond = __ldg(&sm_helper[k].lprob_s_cond_shape);
    float lprob_s_ucond = __ldg(&sm_helper[k].lprob_s_ucond_shape);

    // -- [app] unpack --
    double lprob_f_app = sm_helper[k].lprob_f_app;
    double lprob_k_cond_app = sm_helper[k].lprob_k_cond_app;
    double lprob_s_cond_app = sm_helper[k].lprob_s_cond_app;

    // -- [sum!] --
    lprob_f = lprob_f + lprob_f_app;
    lprob_k_cond = lprob_k_cond + lprob_k_cond_app;
    lprob_s_cond = lprob_s_cond + lprob_s_cond_app;

    // -- determine which is "conditioned" --
    bool select = lprob_k_cond > lprob_s_cond;
    float lprob_sel_cond = select ? lprob_k_cond : lprob_s_cond;
    float lprob_sel_ucond = select ? lprob_s_ucond : lprob_k_ucond;
    sm_helper[k].select = select; // pick "k" if true

    // -- compute hastings [old] --
    // float log_nominator = log_alpha\
    //   + lgammaf(count_k) +  lgammaf(count_s) + lprob_k + lprob_s;
    // float log_denominator = lgammaf(count_f) + lprob_f;
    // sm_helper[k].hasting = log_nominator - log_denominator;

    // -- compute hastings --
    double log_const = lgammaf(count_k) +  lgammaf(count_s) - lgammaf(count_f);
    // // double log_const = 0;
    // // double hastings = log_const + log_alpha + lprob_k + lprob_s - lprob_f;
    // double hastings = log_alpha + lprob_k + lprob_s - lprob_f;

    // -- determine if any splitting --

    // -- [looks good] --
    // float pc2 = sqrt(prior_count_f);
    // float exp_c = 1+exp(-pc2/10.);
    // float new_term = 2*pc2;
    // float new_term = 2*pc2*exp_c/prior_count_f;
    // double raw_sum = lprob_sel_cond + lprob_sel_ucond - lprob_f;
    // double hastings=log_alpha + lprob_sel_cond + lprob_sel_ucond - lprob_f - new_term;
    // double hastings = (lprob_sel_cond > lprob_f) and (lprob_sel_ucond > lprob_f);
    double hastings = log_alpha + lprob_sel_cond + lprob_sel_ucond - lprob_f;

    // hastings = 1;
    // if (count_f > 10000){
    //   printf("[split-tag:%d]: %2.3lf\n",k,hastings);
    // }

    // bool bounds = (abs(mu_s_x - 847)<12) and (abs(mu_s_y - 100) < 90);
    // bool bounds = (abs(mu_s_x - (854-50))<100) and (abs(mu_s_y - 0) < 30);
    // bool bounds = (abs(mu_s_x - (854-0))<100) and (abs(mu_s_y - 0) < 30);
    // if (bounds){
    //   printf("[split-hastings %d]: %2.3lf | %2.3lf %2.3lf %2.3lf\n",k,hastings,lprob_f,lprob_sel_cond,lprob_sel_ucond);
    // }

    // if (count_f > 62500){ // just too big (100x100); always split
    //   hastings = 1;
    // }else if(count_f < 100){ // just too small (10x10); never split
    //   hastings = -1;
    // }
    // hastings = -1.0;

    // -- [bad; too many long cuts] --
    // double hastings = log_alpha + lprob_sel_cond - lprob_f;

    sm_helper[k].hasting = hastings;
    sm_helper[k].merge = (sm_helper[k].hasting > 0);
    sm_helper[s].merge = (sm_helper[k].hasting > 0);

    // -- info --
    // if (k == 100){
    // if (abs(lprob_s_cond_app) > 0.01){
    //       printf("info[%d,%d,%d]: %2.1f | %2.1f %2.1f,%2.1f | [f] %2.4f [k] %2.4f %2.4f [s] %2.4f,%2.4f | %2.2f | %2.2f %2.2f %2.2f %2.2f\n",
    //              k,s,prop_k ? 1 : 0,
    //              prior_count_f,_count_f,count_k,count_s,
    //              lprob_f,
    //              lprob_k_cond,lprob_k_ucond,
    //              lprob_s_cond,lprob_s_ucond,
    //              new_term,
    //              lprob_f_app,lprob_s_cond_app,lprob_k_cond_app,raw_sum);
    //       // printf("info[%d,%d] %lf,%f,%f,%f,%f,%f,%lf\n",
    //       //    k,s,log_const,lprob_f,lprob_k_cond,lprob_s_cond,
    //       //    lprob_k_ucond,lprob_s_ucond,hastings);
    // }

    if((sm_helper[k].merge)){ // split step
    // if(false){

        // printf("info[%d,%d] %lf,%f,%f,%f\n",k,s,log_const,lprob_f,lprob_k,lprob_s);
        // s = atomicAdd(max_sp,1) +1; //
        s = atomicAdd(max_sp,1)+1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;

        // if (count_f > 1312){
        //   printf("[splitting!] info[%d,%d,%d] %f,%f,%f,%f,%f  %f,%f,%f,%f,%f\n",
        //          k,s,prop_k ? 1 : 0,
        //          prior_count_f,count_f,_count_f,count_k,count_s,
        //          lprob_f,lprob_k_cond,lprob_s_cond,lprob_k_ucond,lprob_s_ucond);
        // // printf("[splitting?] info[%d,%d] %f,%f,%lf\n",k,s,count_f,_count_f,hastings);
        // }

        // if (count_f != _count_f){
        //     printf("[splitting!] info[%d,%d,%d] %2.2f,%2.2f,%2.2f,%2.2f,%2.2f  %2.2f,%2.2f,%2.2f,%2.2f,%2.2f\n",k,s,prop_k ? 1 : 0,prior_count_f,count_f,_count_f,count_k,count_s,lprob_f,lprob_k_cond,lprob_s_cond,lprob_k_ucond,lprob_s_ucond);
        //   }

        // if (count_f > 10000){
        //   printf("info[%d,%d] %f,%f\n",k,s,count_f,_count_f);
        // }

        // if (count_f > 1312){
        //   printf("[splitting!] info[%d,%d] %f,%f\n",k,s,count_f,_count_f);
        // }
        // if (true){
        if (false){
          printf("[splitting!] info[%d,%d,%d] %2.1f | %2.1f %2.1f,%2.1f | %2.1f,%2.1f,%2.1f|%2.1f,%2.1f| %2.2f %2.2f %2.2f\n",
                 k,s,prop_k ? 1 : 0,
                 prior_count_f,_count_f,count_k,count_s,
                 lprob_f,lprob_k_cond,lprob_s_cond,
                 lprob_k_ucond,lprob_s_ucond,
                 lprob_f_app,lprob_s_cond_app,lprob_k_cond_app);
        }


        // -- init new spix --
        // float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        // sp_params[k].prior_count = prior_count;
        // sp_params[s].prior_count = prior_count;

        

        // bool select = lprob_k_cond > lprob_s_cond;
        // float lprob_sel_cond = select ? lprob_k_cond : lprob_s_cond;
        // float lprob_sel_ucond = select ? lprob_s_ucond : lprob_k_ucond;
        // sm_helper[k].select = select; // pick "k" if true

        // -- update prior counts --
        float sp_size2 = 1.*sp_size*sp_size;
        // bool prop_k = sp_params[k].prop;
        float prior_count = sp_params[k].prior_count;
        float prior_count_half = max(prior_count/2.0,36.0);
        if(prop_k){
          // sp_params[k].prior_count = prior_count; // old
          sp_params[k].prior_count = prior_count;
          sp_params[s].prior_count = prior_count/2.;
          // sp_params[k].prior_count = select ? prior_count : prior_count_half;
          // sp_params[s].prior_count = select ? prior_count_half : prior_count;
          // sp_params[k].prior_count = select ? prior_count : sp_size2;
          // sp_params[s].prior_count = select ? sp_size2 : prior_count;
          // sp_params[k].prior_count = select ? prior_count_half : sp_size2;
          // sp_params[s].prior_count = select ? sp_size2 : prior_count_half;
          // sp_params[k].prior_count = prior_count_half;
          // sp_params[s].prior_count = prior_count_half;
          // sp_params[k].prop = false;//select;
          // sp_params[s].prop = false;//not(select);
          // sp_params[k].prop = select;
          // sp_params[s].prop = not(select);
          sp_params[s].prop = false;
        }else{
          sp_params[k].prior_count = prior_count_half;
          sp_params[s].prior_count = prior_count_half;
          sp_params[s].prop = false;

          double3 prior_icov;
          prior_icov.x = sp_params[s].prior_count;
          prior_icov.y = 0;
          prior_icov.z = sp_params[s].prior_count;
          sp_params[k].prior_icov = prior_icov;

        }

        double3 prior_icov;
        sp_params[k].icount = sp_params[k].icount/2.0;
        prior_icov.x = sp_params[s].prior_count;
        prior_icov.y = 0;
        prior_icov.z = sp_params[s].prior_count;
        sp_params[s].prior_icov = prior_icov;
        sp_params[s].valid = 1;

        // printf("[%d,%d]: %2.0f %2.0f %2.0f | %2.3lf %2.3lf %2.3lf\n",
        //        k,s,count_f,count_s,count_k,lprob_f,lprob_s_cond,lprob_k_cond);

        // double3 prior_sigma_shape;
        // prior_sigma_shape.x = 1./sp_size;
        // prior_sigma_shape.y = 0;
        // prior_sigma_shape.z = 1./sp_size;
        // sp_params[s].prior_sigma_shape = prior_sigma_shape;
        
        // double2 prior_mu_shape;
        // prior_mu_shape.x = 0;
        // prior_mu_shape.y = 0;
        // sp_params[s].prior_mu_shape = prior_mu_shape;

        // // -- [appearance] prior --
        // float3 mu_prior;
        // mu_prior.x = 0;
        // mu_prior.y = 0;
        // mu_prior.z = 0;
        // sp_params[s].prior_mu_app = mu_prior;
        // sp_params[s].prior_mu_app_count = 1;

        // // -- [shape] prior --
        // double2 prior_mu_shape;
        // prior_mu_shape.x = 0;
        // prior_mu_shape.y = 0;
        // sp_params[s].prior_mu_shape = prior_mu_shape;
        // sp_params[s].prior_mu_shape_count = 1;
        // double3 prior_sigma_shape;
        // prior_sigma_shape.x = prior_count;
        // prior_sigma_shape.y = 0;
        // prior_sigma_shape.z = prior_count;
        // sp_params[s].prior_sigma_shape = prior_sigma_shape;
        // sp_params[s].prior_sigma_shape_count = prior_count;
        // sp_params[s].prop = false;

      }
    // else if (count_f > 300){
    //     // printf("[%d,%d]: %2.0f %2.0f %2.0f | %2.3lf %2.3lf %2.3lf\n",
    //     //        k,s,count_f,count_s,count_k,lprob_f,lprob_s_cond,lprob_k_cond);
    // }

}

__global__ void merge_sp_p(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int k = seg[idx]; // center 
    //if (sp_params[k].valid == 0) return;
    int f = sm_pairs[2*k+1];
    if(sm_helper[k].remove){
      bool prop = sp_params[k].prop;
      assert(prop==false);
      bool valid = sp_params[f].valid;
      assert(valid == true);
      assert(f>=0);
      seg[idx] =  f;
      // printf("%d,%d\n",k,f);
    }

    return;  
      
}

__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_spix){   

  // todo: add nbatch, no sftrs
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=npix) return; 
    int k = seg[idx]; // center 
    int k2 = k + (max_spix + 1);
    if (sp_params[k].valid == 0){ return; }
    if ((sm_helper[k].merge == false)||(sm_helper[k2].merge == false)){
      return;
    }

    int s = sm_pairs[2*k];
    if (s < 0){ return; }
    
    if (sm_helper[k].select){
      if(sm_seg1[idx]==k2) {
        assert(s>=0);
        seg[idx] = s;
      }
    }else{
      if(sm_seg1[idx]==k) {
        assert(s>=0);
        seg[idx] = s;
      }
    }

    return;  
}



__global__ void remove_sp_p(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                            const int nspix_buffer, int* nmerges) {

	// -- getting the index of the pixel --
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
    int s = sm_pairs[2*k+1];
    if(s<0) return;
    bool is_cycle = sm_pairs[2*s+1] == k;
    if ((sp_params[k].valid == 0)||(sp_params[s].valid == 0)) return;    
    // if ((sm_helper[k].merge == true) && (sm_helper[f].merge == false) && (split_merge_pairs[2*f]==k) )
    // if ((sm_helper[k].merge==true)&&(sm_helper[s].merge==false)&&(sm_pairs[2*s]==k))
    // if ((sm_helper[k].merge == true) && (sm_helper[s].merge == false))
    bool is_prop_k = sp_params[k].prop;
    bool is_prop_s = sp_params[s].prop;
    assert(not(is_prop_k and is_prop_s));
    bool no_prop = not(is_prop_k) and not(is_prop_s);
    // bool cond_c = is_cycle and (is_prop_s or ((s<k) and no_prop)); // allows small merge?

    // if its a cycle, we just have to pick one
    // if the pair is (prop,no-prop), we merge the "no-prop" into "prop"
    // if the pair is (no-prop,no-prop), we pick the smaller value
    // bool cond_c = is_cycle and (is_prop_s or ((s<k) and no_prop)); // allows small merge?
    bool cond_c = is_cycle and (is_prop_k or (k<s)); // allows small merge?
    // bool cond_c = is_cycle and is_prop_k;

    if((sm_helper[k].merge==true)&&((sm_helper[s].merge==false)||cond_c))
      {

        // -- keep the propogated --
        // assert(not(is_prop_k and is_prop_s));
        if (is_prop_k){
          sm_pairs[2*s+1] = k; // "thread k" is the ONLY writer to 2*s+1
          int tmp = k;
          k = s;
          s = tmp;
        }
        // int f = sm_pairs[2*k+1];

        // atomicAdd(nmerges,1);
        sm_helper[k].remove=true;
        sp_params[k].valid = 0;

        // -- update priors --
        sp_params[s].prior_count =sp_params[k].prior_count+sp_params[s].prior_count;
        // sp_params[s].prior_sigma_shape.x = 1.0/sp_params[s].prior_count;
        // sp_params[s].prior_sigma_shape.y = 0.;
        // sp_params[s].prior_sigma_shape.z = 1.0/sp_params[s].prior_count;

        // float pc_k = sp_params[k].prior_count;
        // float pc_s = sp_params[s].prior_count;
        // double3 scov_k = sp_params[k].sample_sigma_shape;
        // double3 scov_s = sp_params[s].sample_sigma_shape;
        // double3 scov_ks;
        // scov_ks.x = scov_k.x*pc_k + ;
        // prior_icov

        // todo! update the covariance if propogated
        // // -- update prior counts --
        // // float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        // float sp_size2 = 1.0*sp_size*sp_size;
        // // bool prop_k = sp_params[k].prop;
        // float prior_count = sp_params[k].prior_count;
        // float prior_count_half = max(prior_count/2.0,36.0);
        // if(prop_k){
        //   sp_params[k].prior_count = prior_count;
        //   sp_params[s].prior_count = prior_count/2.;
        //   sp_params[s].prop = false;
        // }else{
        //   sp_params[k].prior_count = prior_count_half;
        //   sp_params[s].prior_count = prior_count_half;
        //   sp_params[s].prop = false;

        //   double3 prior_icov;
        //   prior_icov.x = sp_params[s].prior_count;
        //   prior_icov.y = 0;
        //   prior_icov.z = sp_params[s].prior_count;
        //   sp_params[k].prior_icov = prior_icov;

        // }

        // double3 prior_icov;
        // sp_params[k].icount = sp_params[k].icount/2.0;
        // prior_icov.x = sp_params[s].prior_count;
        // prior_icov.y = 0;
        // prior_icov.z = sp_params[s].prior_count;
        // sp_params[s].prior_icov = prior_icov;
        // sp_params[s].valid = 1;

        // // -> this suggests we should maintain sample estimates across time
        // //    since this can be easily updated in these loops.
        // //    i suspect we are already doing this.


      }
    // else
    //   {
    //     sm_helper[k].remove=false;
    //   }
    
    return;
    
}



__device__ longlong2 get_sum_shape(longlong2 sum_s, longlong2 sum_k){
  longlong2 sum_f;
  sum_f.x = sum_s.x+sum_k.x;
  sum_f.y = sum_s.y+sum_k.y;
  return sum_f;
}

__device__ longlong3 get_sq_sum_shape(longlong3 sq_sum_s, longlong3 sq_sum_k){
  longlong3 sq_sum_f;
  sq_sum_f.x = sq_sum_s.x+sq_sum_k.x;
  sq_sum_f.y = sq_sum_s.y+sq_sum_k.y;
  sq_sum_f.z = sq_sum_s.z+sq_sum_k.z;
  return sq_sum_f;
}


__device__ double3 add_sigma_smoothing(double3 in_sigma, int count,
                                       float pc, int sp_size) {
  // -- sample cov --
  int nf = 50;
  double total_count = 1.*(count + pc*nf);
  double3 sigma;
  sigma.x = (pc*sp_size + count*in_sigma.x)/(total_count + 3.0);
  sigma.y = (count*in_sigma.y) / (total_count + 3.0);
  sigma.z = (pc*sp_size + count*in_sigma.z)/(total_count + 3.0);
  return sigma;
  // return in_sigma;
}

__device__ double3 compute_sigma_shape(longlong2 sum, longlong3 sq_sum,
                                       int _count, float prior_count, int sp_size) {

  // -- mean --
  double count = 1.0*_count;
  double2 mu;
  mu.x = sum.x/count;
  mu.y = sum.y/count;
  
  // -- sample covariance --
  double3 sigma;
  sigma.x = sq_sum.x/count - (mu.x * mu.x);
  sigma.y = sq_sum.y/count - (mu.x * mu.y);
  sigma.z = sq_sum.z/count - (mu.y * mu.y);

  return add_sigma_smoothing(sigma,_count,prior_count,sp_size);
  // // // -- info --
  // // printf("sxx,sxy,syy,pc: %lf %lf %lf %f %lf | %d %d | %lld %lld %lld\n",
  // //        sigma.x,sigma.y,sigma.z,prior_count,count,
  // //        sum.x,sum.y,sq_sum.x,sq_sum.y,sq_sum.z);

  // // -- sample cov --
  // int nf = 50;
  // float pc = prior_count;
  // // double total_count = 1.*(count + prior_count);
  // double total_count = 1.*(count + pc*nf);
  // sigma.x = (pc*sp_size + sigma.x)/(total_count + 3.0);
  // sigma.y = (sigma.y) / (total_count + 3.0);
  // sigma.z = (pc*sp_size + sigma.z)/(total_count + 3.0);

  // // -- determinant --
  // // double det = sigma.x*sigma.z - sigma.y*sigma.y;
  // // if (det < 0.0001){ det = 1.; }
  // // double tmp;
  // // tmp = sigma.x;
  // // sigma.x = sigma.z/det;
  // // sigma.y = -sigma.y/det;
  // // sigma.z = tmp/det;

  // // double3 isigma;
  // // isigma.x = sigma.z/det;
  // // isigma.y = -sigma.y/det;
  // // isigma.z = sigma.x/det;
  // // return isigma;

  // return sigma;
}
































/************************************************************



                       Merge Functions



*************************************************************/

// old name: calc_bn(int* seg
__global__ void calc_merge_stats_step0_p(int* sm_pairs,
                                       spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer, float b_0) {

    // todo -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    int f = sm_pairs[2*k+1];
	//if (sp_params[f].valid == 0) return;
    // if (f<=0) return;
    if (f<0) return;

    // -- read --
    float count_f = __ldg(&sp_params[f].count);
    float count_k = __ldg(&sp_params[k].count);

    float squares_f_x = __ldg(&sm_helper[f].sq_sum_app.x);
    float squares_f_y = __ldg(&sm_helper[f].sq_sum_app.y);
    float squares_f_z = __ldg(&sm_helper[f].sq_sum_app.z);
   
    float squares_k_x = __ldg(&sm_helper[k].sq_sum_app.x);
    float squares_k_y = __ldg(&sm_helper[k].sq_sum_app.y);
    float squares_k_z = __ldg(&sm_helper[k].sq_sum_app.z);
   
    float mu_f_x = __ldg(&sp_helper[f].sum_app.x);
    float mu_f_y = __ldg(&sp_helper[f].sum_app.y);
    float mu_f_z = __ldg(&sp_helper[f].sum_app.z);
   
    float mu_k_x = __ldg(&sp_helper[k].sum_app.x);
    float mu_k_y = __ldg(&sp_helper[k].sum_app.y);
    float mu_k_z = __ldg(&sp_helper[k].sum_app.z);
    int count_fk = count_f + count_k;

    // -- compute summary stats --
    sm_helper[k].count = count_fk;
    sm_helper[k].b_n_app.x = b_0 + 0.5 * ((squares_k_x) - (mu_k_x*mu_k_x/count_k));
    
    sm_helper[k].b_n_f_app.x = b_0 + \
      0.5 *( (squares_k_x+squares_f_x) - ( (mu_f_x + mu_k_x ) * (mu_f_x + mu_k_x ) / (count_fk)));

    sm_helper[k].b_n_app.y = b_0 + 0.5 * ((squares_k_y) - (mu_k_y*mu_k_y/count_k));
    
    sm_helper[k].b_n_f_app.y = b_0 + \
      0.5 *( (squares_k_y+squares_f_y) - ((mu_f_y + mu_k_y ) * (mu_f_y + mu_k_y ) / (count_fk)));

    sm_helper[k].b_n_app.z = b_0 + 0.5 * ((squares_k_z) - (mu_k_z*mu_k_z/count_k));
    
    sm_helper[k].b_n_f_app.z = b_0 + \
      0.5 *( (squares_k_z+squares_f_z) - ( (mu_f_z + mu_k_z ) * (mu_f_z + mu_k_z ) / (count_fk)));

    if(  sm_helper[k].b_n_app.x<0)   sm_helper[k].b_n_app.x = 0.1;
    if(  sm_helper[k].b_n_app.y<0)   sm_helper[k].b_n_app.y = 0.1;
    if(  sm_helper[k].b_n_app.z<0)   sm_helper[k].b_n_app.z = 0.1;

    if(  sm_helper[k].b_n_f_app.x<0)   sm_helper[k].b_n_f_app.x = 0.1;
    if(  sm_helper[k].b_n_f_app.y<0)   sm_helper[k].b_n_f_app.y = 0.1;
    if(  sm_helper[k].b_n_f_app.z<0)   sm_helper[k].b_n_f_app.z = 0.1;

}

__global__
void calc_merge_stats_step1_p(spix_params* sp_params,spix_helper_sm_v2* sm_helper,
                            const int nspix_buffer,float a_0, float b_0) {

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- read --
    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count);
    float a_n = a_0 + float(count_k) / 2;
    float a_n_f = a_0+ float(count_f) / 2;
    float v_n = 1/float(count_k);
    float v_n_f = 1/float(count_f);


    // -- update --
    a_0 = a_n;
    sm_helper[k].numerator_app = a_0 * __logf(b_0) + lgammaf(a_n)+0.5*__logf(v_n);
    sm_helper[k].denominator.x = a_n* __logf ( __ldg(&sm_helper[k].b_n_app.x)) + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator.y = a_n* __logf ( __ldg(&sm_helper[k].b_n_app.y)) + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgamma(a_0);
    sm_helper[k].denominator.z = a_n* __logf(__ldg(&sm_helper[k].b_n_app.z)) \
      + 0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);
    
    a_0 = a_n_f;
    sm_helper[k].numerator_f_app = a_0 * __logf (b_0) + lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sm_helper[k].denominator_f.x = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.x)) + 0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator_f.y = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.y)) + 0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);
    sm_helper[k].denominator_f.z = a_n_f* __logf (__ldg(&sm_helper[k].b_n_f_app.z)) + 0.5 * count_f* __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);         

}   

// old name: calc_hasting_ratio(const float* image
__global__ void calc_merge_stats_step2_p(int* sm_pairs,
                                       spix_params* sp_params,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                         float alpha, float merge_alpha) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int f = sm_pairs[2*k+1];
    // printf("%d,%d\n",k,f);

    if(f<0) return;
    // printf("%d,%d\n",f,sp_params[f].valid == 0 ? 1: 0);
	if (sp_params[f].valid == 0) return;
    // if(f<=0) return;


    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count);
    
    if ((count_k<1)||(count_f<1)) return;

    sm_helper[k].merge = false;
    float num_k = __ldg(&sm_helper[k].numerator_app);

    float total_marginal_1 = (num_k - __ldg(&sm_helper[k].denominator.x)) +  
                         (num_k - __ldg(&sm_helper[k].denominator.y)) + 
                         (num_k - __ldg(&sm_helper[k].denominator.z)); 

    float num_f = __ldg(&sm_helper[f].numerator_app);

    float total_marginal_2 = (num_f - __ldg(&sm_helper[f].denominator.x)) +   
                         (num_f - __ldg(&sm_helper[f].denominator.y)) + 
                         (num_f - __ldg(&sm_helper[f].denominator.z));

    float num_kf = __ldg(&sm_helper[k].numerator_f_app);

    float total_marginal_f = (num_kf - __ldg(&sm_helper[k].denominator_f.x)) +   
                         (num_kf - __ldg(&sm_helper[k].denominator_f.y)) + 
                         (num_kf - __ldg(&sm_helper[k].denominator_f.z));


    float log_nominator = lgammaf(count_f) + total_marginal_f + lgammaf(alpha) + 
        lgammaf(alpha / 2 + count_k) + lgammaf(alpha / 2 + count_f -  count_k);

   float log_denominator = __logf(alpha) + lgammaf(count_k) + lgammaf(count_f -  count_k) + total_marginal_1 + 
        total_marginal_2 + lgammaf(alpha + count_f) + lgammaf(alpha / 2) + 
        lgammaf(alpha / 2);

    log_denominator = __logf(alpha) + total_marginal_1 + total_marginal_2;
    log_nominator = total_marginal_f ;


    sm_helper[k].hasting = log_nominator - log_denominator + merge_alpha;

    // printf("[%2.2f,%2.2f]: %2.2f,%2.2f,%2.2f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,sm_helper[k].hasting,log_nominator,log_denominator);

    return;
}


__global__ void update_merge_flag_p(int* sm_pairs, spix_params* sp_params,
                                  spix_helper_sm_v2* sm_helper, const int nspix_buffer,
                                  int* nmerges) {

	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = sm_pairs[2*k+1];
    // if(f<=0) return;
    if(f<0) return;
	if (sp_params[f].valid == 0) return;

    // -- determine which is "propogated" --
    bool is_prop_k = sp_params[k].prop;
    bool is_prop_f = sp_params[f].prop;
    assert(not(is_prop_k and is_prop_f)); // not both propogated.
    // bool propogate_k = (ninvalid_k <= ninvalid_f);
    // sm_helper[k].select = propogate_k or not(is_prop); // pick "k" if true
    // bool prop_k = propogate_k;


    if((sm_helper[k].hasting ) > -2)
    {
      //printf("Want to merge k: %d, f: %d, splitmerge k %d, splitmerge  f %d, %d\n", k, f, sm_pairs[2*k], sm_pairs[2*f], sm_pairs[2*f+1] );
      int curr_max = atomicMax(&sm_pairs[2*f],k);
      if( curr_max == -1){
        // atomicAdd(nmerges,1);
        sm_helper[k].merge = true;
      }
      // else{ // idk why I included this...
      //   sm_pairs[2*f] = curr_max;
      // }

      // int curr_max = atomicMax(&sm_pairs[2*f],k);
      // if( curr_max == 0){
      //   //printf("Merge: %f \n",sm_helper[k].hasting );
      //   sm_helper[k].merge = true;
      // }else{
      //   sm_pairs[2*f] = curr_max;
      // }

    }
         
    return;

}

































































/************************************************************



                       Split Functions



*************************************************************/


// old name: calc_bn_split
__global__ void calc_split_stats_step0_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                       float b_0, int max_spix) {
  // todo; -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    //get the label of neigh
    // int s = k + max_SP;

    int s = k + (max_spix+1);
	if (s>=nspix_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k= __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);
    if((count_f<1)||( count_k<1)||(count_s<1)) return;

    // -- read params --
    float squares_s_x = __ldg(&sm_helper[s].sq_sum_app.x);
    float squares_s_y = __ldg(&sm_helper[s].sq_sum_app.y);
    float squares_s_z = __ldg(&sm_helper[s].sq_sum_app.z);
   
    float squares_k_x = __ldg(&sm_helper[k].sq_sum_app.x);
    float squares_k_y = __ldg(&sm_helper[k].sq_sum_app.y);
    float squares_k_z = __ldg(&sm_helper[k].sq_sum_app.z);
   
    float mu_s_x = __ldg(&sm_helper[s].sum_app.x);
    float mu_s_y = __ldg(&sm_helper[s].sum_app.y);
    float mu_s_z = __ldg(&sm_helper[s].sum_app.z);

    float mu_k_x = __ldg(&sm_helper[k].sum_app.x);
    float mu_k_y = __ldg(&sm_helper[k].sum_app.y);
    float mu_k_z = __ldg(&sm_helper[k].sum_app.z);

    float mu_f_x = __ldg(&sp_helper[k].sum_app.x);
    float mu_f_y = __ldg(&sp_helper[k].sum_app.y);
    float mu_f_z = __ldg(&sp_helper[k].sum_app.z);

    // printf("mu_s_x,mu_k_x,mu_f_x: %2.2f,%2.2f,%2.2f\n",mu_s_x,mu_k_x,mu_f_x);

    // -- check location --
    // printf("[%2.2f,%2.2f]: %2.2f+%2.2f  = %2.2f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,mu_s_x,mu_k_x,mu_f_x);

    // -- compute summary stats --
    sm_helper[k].b_n_app.x = b_0 + 0.5 * ((squares_k_x) -
                                ( (mu_k_x*mu_k_x)/ (count_k)));
    sm_helper[k].b_n_app.y = b_0 + 0.5 * ((squares_k_y) -
                                ( mu_k_y*mu_k_y/ count_k));
    sm_helper[k].b_n_app.z = b_0 + 0.5 * ((squares_k_z) -
                                ( mu_k_z*mu_k_z/ count_k));
 
    sm_helper[s].b_n_app.x = b_0 + 0.5 * ((squares_s_x) -
                                ( mu_s_x*mu_s_x/ count_s));
    sm_helper[s].b_n_app.y = b_0 + 0.5 * ((squares_s_y) -
                                ( mu_s_y*mu_s_y/ count_s));
    sm_helper[s].b_n_app.z = b_0 + 0.5 * ((squares_s_z) -
                                ( mu_s_z*mu_s_z/ count_s));

    sm_helper[k].b_n_f_app.x = b_0 + 0.5 * ((squares_k_x+squares_s_x) -
                                ( mu_f_x*mu_f_x/ count_f));
    sm_helper[k].b_n_f_app.y = b_0 + 0.5 * ((squares_k_y+squares_s_y) -
                                ( mu_f_y*mu_f_y/ count_f));
    sm_helper[k].b_n_f_app.z = b_0 + 0.5 * ((squares_k_z+squares_s_z) -
                                ( mu_f_z*mu_f_z/ count_f));
                       
}

// old name: calc_marginal_liklelyhoood_of_sp_split
__global__ void calc_split_stats_step1_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                       float a_0, float b_0, int max_spix) {

  // todo -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nspix_buffer) return;
    int s = k + (max_spix+1);
    if (s>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);

    if((count_f<1)||( count_k<1)||(count_s<1)) return;
    // if (count_f != (count_k+count_s)){
    //   printf("count_f,count_k,count_s: %f,%f,%f\n",count_f,count_k,count_s);
    // }
    if (count_f!=count_k+count_s) return;
    // assert(count_f == (count_k+count_s));
    // if (count_f!=count_k+count_s) return;
    // TODO: check if there is no neigh
    // TODO: check if num is the same
	//get the label
    //a_0 = 1100*(count_f);

    float a_n_k = a_0+float(count_k)/2;
    float a_n_s = a_0+float(count_s)/2;
    float a_n_f = a_0+float(count_f)/2;


    float v_n_k = 1/float(count_k);
    float v_n_s = 1/float(count_s);
    float v_n_f = 1/float(count_f);
   /* v_n_k = 1;
    v_n_f =1;
    v_n_s=1;*/

    float b_n_k_x = __ldg(&sm_helper[k].b_n_app.x);
    float b_n_k_y = __ldg(&sm_helper[k].b_n_app.y);
    float b_n_k_z = __ldg(&sm_helper[k].b_n_app.z);

    float b_n_s_x = __ldg(&sm_helper[s].b_n_app.x);
    float b_n_s_y = __ldg(&sm_helper[s].b_n_app.y);
    float b_n_s_z = __ldg(&sm_helper[s].b_n_app.z);

    float b_n_f_app_x = __ldg(&sm_helper[k].b_n_f_app.x);
    float b_n_f_app_y = __ldg(&sm_helper[k].b_n_f_app.y);
    float b_n_f_app_z = __ldg(&sm_helper[k].b_n_f_app.z);


    a_0 = a_n_k;
    sm_helper[k].numerator_app = a_0 * __logf(b_0) + \
      lgammaf(a_n_k)+ 0.5*__logf(v_n_k);
    sm_helper[k].denominator.x = a_n_k * __logf (b_n_k_x) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator.y = a_n_k * __logf (b_n_k_y) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator.z = a_n_k * __logf (b_n_k_z) + \
      0.5 * count_k * __logf (M_PI) + count_k * __logf (2) + lgammaf(a_0);

    a_0 = a_n_s;
    sm_helper[s].numerator_app = a_0 * __logf(b_0) + \
      lgammaf(a_n_s)+0.5*__logf(v_n_s);
    sm_helper[s].denominator.x = a_n_s * __logf (b_n_s_x) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);
    sm_helper[s].denominator.y = a_n_s * __logf (b_n_s_y) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);
    sm_helper[s].denominator.z = a_n_s * __logf (b_n_s_z) + \
      0.5 * count_s * __logf (M_PI) + count_s * __logf (2) + lgammaf(a_0);      

    a_0 =a_n_f;
    sm_helper[k].numerator_f_app =a_0*__logf(b_0)+lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sm_helper[k].denominator_f.x = a_n_f * __logf (b_n_f_app_x) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator_f.y = a_n_f * __logf (b_n_f_app_y) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);

    sm_helper[k].denominator_f.z = a_n_f * __logf (b_n_f_app_z) + \
      0.5 * count_f * __logf (M_PI) + count_f * __logf (2) + lgammaf(a_0);        

}   



__global__
void update_split_flag_p(int* sm_pairs, spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper, const int nspix_buffer,
                         float alpha, float split_alpha, float iperc_coeff,
                         int sp_size, int max_spix, int* max_sp) {
  
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    int s = k + (max_spix+1);
    if(s>=nspix_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k = __ldg(&sm_helper[k].count);
    float count_s = __ldg(&sm_helper[s].count);

    if((count_f<1)||(count_k<1)||(count_s<1)) return;
    // if (count_f != (count_k+count_s)){
    //   printf("[split_flag@%d] count_f,count_k,count_s: %f,%f,%f\n",
    //          k,count_f,count_k,count_s);
    // }
    if (count_f!=count_k+count_s) return;


    // -- determine which is "propogated" --
    bool is_prop = sp_params[k].prop;
    int ninvalid_k = sm_helper[k].ninvalid;
    int ninvalid_s = sm_helper[s].ninvalid;
    bool propogate_k = (ninvalid_k <= ninvalid_s);
    bool prop_k = propogate_k;
    sm_helper[k].select = propogate_k or not(is_prop); // pick "k" if true
    float iperc_k = ninvalid_k/(1.0*count_k);
    // float iperc_s = ninvalid_s/(1.0*count_s);
    float iperc_s = ninvalid_s/(1.0*count_s);
    float iperc_prop = propogate_k ? iperc_k : iperc_s;
    // iperc_prop = is_prop ? iperc_prop : 1.0;

    // -- .. --
    float num_k = __ldg(&sm_helper[k].numerator_app);
    float num_s = __ldg(&sm_helper[s].numerator_app);
    float num_f = __ldg(&sm_helper[k].numerator_f_app);
    float total_marginal_k = (num_k - __ldg(&sm_helper[k].denominator.x)) +  
                         (num_k - __ldg(&sm_helper[k].denominator.y)) + 
                         (num_k - __ldg(&sm_helper[k].denominator.z)); 
    float total_marginal_s = (num_s - __ldg(&sm_helper[s].denominator.x)) +  
                         (num_s - __ldg(&sm_helper[s].denominator.y)) + 
                         (num_s - __ldg(&sm_helper[s].denominator.z)); 
    float total_marginal_f = (num_f - __ldg(&sm_helper[k].denominator_f.x)) +  
                         (num_f - __ldg(&sm_helper[k].denominator_f.y)) + 
                         (num_f - __ldg(&sm_helper[k].denominator_f.z)); 

 
     //printf("hasating:x k: %d, count: %f, den: %f, %f, %f, b_n: %f, %f, %f, num: %f \n",k, count_k,  sm_helper[k].denominator.x, sm_helper[k].denominator.y,  sm_helper[k].denominator.z,   __logf (sm_helper[k].b_n_app.x) ,  __logf (sm_helper[k].b_n_app.y),   __logf (sm_helper[k].b_n_app.z), sm_helper[k].numerator_app);

    float log_nominator = __logf(alpha)+ lgammaf(count_k)\
      + total_marginal_k + lgammaf(count_s) + total_marginal_s;
    log_nominator = total_marginal_k + total_marginal_s;

    float log_denominator = lgammaf(count_f) + total_marginal_f; // ?? what is this line for?
    log_denominator =total_marginal_f;


    // float hastings = log_nominator - log_denominator + split_alpha - 10*(1.0-iperc);
    // float hastings = log_nominator - log_denominator + split_alpha - 5*(1.0-iperc);
    float hastings=log_nominator - log_denominator +split_alpha-iperc_coeff*(1.0-iperc_prop);
    // hastings = iperc*hastings + -10*(1.0-iperc);
    // sm_helper[k].hasting = log_nominator - log_denominator + split_alpha;
    // sm_helper[k].hasting = log_nominator - log_denominator + split_alpha - 2*(1.0-iperc);

    // -- no split if too small --
    if (count_f < 64){
      hastings = -10;
    }

    // -- check location --
    // printf("[%2.2f,%2.2f]: %2.3f, %2.3f, %2.3f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,
    //        sm_helper[k].hasting,log_nominator,log_denominator);
    // printf("[%2.2f,%2.2f]: %2.2f+%2.2f  = %2.2f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,mu_s_x,mu_k_x,mu_f_x);
    // printf("hasting: %2.3f, %2.3f, %2.3f\n",
    //        sm_helper[k].hasting,log_nominator,log_denominator);

    // ".merge" is merely a bool variable; nothing about merging here. only splitting
    sm_helper[k].hasting = hastings;
    sm_helper[k].merge = (sm_helper[k].hasting > -2); // why "-2"?
    sm_helper[s].merge = (sm_helper[k].hasting > -2);

    if((sm_helper[k].merge)) // split step
      {

        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;


        // -- update prior counts --
        // float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        float sp_size2 = 1.0*sp_size*sp_size;
        // bool prop_k = sp_params[k].prop;
        float prior_count = sp_params[k].prior_count;
        float prior_count_half = max(prior_count/2.0,36.0);
        if(prop_k){
          // sp_params[k].prior_count = prior_count/2.;
          // sp_params[s].prior_count = prior_count/2.;
          sp_params[k].prior_count = prior_count;
          sp_params[s].prior_count = prior_count_half;
          sp_params[s].prop = false;
        }else{
          sp_params[k].prior_count = prior_count_half;
          sp_params[s].prior_count = prior_count_half;
          sp_params[s].prop = false;

          double3 prior_icov;
          prior_icov.x = sp_params[s].prior_count;
          prior_icov.y = 0;
          prior_icov.z = sp_params[s].prior_count;
          sp_params[k].prior_icov = prior_icov;

        }

        double3 prior_icov;
        sp_params[k].icount = sp_params[k].icount/2.0;
        prior_icov.x = sp_params[s].prior_count;
        prior_icov.y = 0;
        prior_icov.z = sp_params[s].prior_count;
        sp_params[s].prior_icov = prior_icov;
        sp_params[s].valid = 1;


      }

}























