
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

#include "split_merge.h"
#include "split_merge_prop_v2.h"


#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_prop_v2(const float* img, int* seg, bool* border,
              spix_params* sp_params, spix_helper* sp_helper,
              spix_helper_sm* sm_helper,
              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
              float alpha_hastings, float sigma2_app, float sigma2_size,
              int& count, int idx, int max_spix, 
              const int sp_size, 
              const int npix, const int nbatch,
              const int width, const int height,
              const int nftrs, const int nspix_buffer){

  if(idx%4 == 0){
    count += 1;
    // count = 2; // remove me!
    int direction = count%2+1;
    // -- run split --
    max_spix = CudaCalcSplitCandidate_prop_v2(img, seg, border,
                                       sp_params, sp_helper, sm_helper,
                                       sm_seg1, sm_seg2, sm_pairs,
                                       sp_size,npix,nbatch,width,height,nftrs,
                                       nspix_buffer, max_spix,
                                       direction, alpha_hastings,
                                       sigma2_app, sigma2_size);

  }
  return max_spix;
}

__host__
void run_merge_prop_v2(const float* img, int* seg, bool* border,
               spix_params* sp_params, spix_helper* sp_helper,
               spix_helper_sm* sm_helper,
               int* sm_seg1, int* sm_seg2, int* sm_pairs,
               float alpha_hastings,
               float sigma2_app, float sigma2_size,
               int& count, int idx, int max_spix,
               const int sp_size, const int npix, const int nbatch,
               const int width, const int height,
               const int nftrs, const int nspix_buffer){

  if( idx%4 == 2){
    // -- run merge --
    int direction = count%2;
    CudaCalcMergeCandidate_prop_v2(img, seg, border,
                           sp_params, sp_helper, sm_helper, sm_pairs,
                           sp_size,npix,nbatch,width,height,nftrs,
                           nspix_buffer,direction, alpha_hastings,
                           sigma2_app, sigma2_size);

  }
}

__host__ void CudaCalcMergeCandidate_prop_v2(const float* img, int* seg, bool* border,
                                     spix_params* sp_params,spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,int* sm_pairs,
                                     const int sp_size,
                                     const int npix, const int nbatch,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float log_alpha,
                                     float sigma2_app, float sigma2_size){

    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // float alpha = alpha;
    // float a_0 = 10000;
    float a_0 = 1e4;
    // float a_0 = 1e5;
    // float a_0 = 1e6;
    // float b_0 = sigma2_app * (a_0) ;
    float i_std = 0.018;
    // i_std = sigma2_app;
    i_std = sqrt(sigma2_app)*2;
    // printf("i_std,sigma2_app: %2.4f,%2.4f\n",i_std,sigma2_app);
    float b_0 = i_std * (a_0) ;
    float alpha = exp(log_alpha);

    int nvalid_cpu;
    int* nvalid;
    cudaMalloc((void **)&nvalid, sizeof(int));
    cudaMemset(nvalid, 0,sizeof(int));

    int nmerges;
    int* nmerges_gpu;
    cudaMalloc((void **)&nmerges_gpu, sizeof(int));
    cudaMemset(nmerges_gpu, 0,sizeof(int));

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                              nspix_buffer, nbatch, width,
                                              nftrs, npix, sm_pairs, nvalid);


    // fprintf(stdout,"direction: %d\n",direction);
    calc_merge_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                          npix, nbatch, width,
                                                          height, direction); 

    sum_by_label<<<BlockPerGrid,ThreadPerBlock>>>(img,seg,sp_params,sm_helper,
                                                        npix, nbatch, width,  nftrs);


    // -- summary stats of merge --
    calc_merge_stats_step0_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                             sp_helper,sm_helper,
                                                             nspix_buffer, b_0);

    calc_merge_stats_step1_p<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sm_helper,
                                                             nspix_buffer,a_0, b_0);

    calc_merge_stats_step2_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                             sm_helper,nspix_buffer,
                                                             alpha);


    // -- update the merge flag: "to merge or not to merge" --
    update_merge_flag_p<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                        sm_helper,nspix_buffer,
                                                        nmerges_gpu);


    // -- count number of merges --
    cudaMemcpy(&nmerges,nmerges_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("nmerges: %d\n",nmerges);
    cudaMemset(nmerges_gpu, 0,sizeof(int));
    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[merge] nvalid: %d\n",nvalid_cpu);



    // -- actually merge --
    remove_sp<<<BlockPerGrid2,ThreadPerBlock>>>(sm_pairs,sp_params,
                                                sm_helper,nspix_buffer);


    merge_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs, sp_params,
                                              sm_helper, npix, nbatch, width, height);  


    // -- free! --
    cudaFree(nvalid);
    cudaFree(nmerges_gpu);


}


__host__ int CudaCalcSplitCandidate_prop_v2(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int sp_size,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_spix,
                                    int direction, float log_alpha,
                                    float sigma2_app, float sigma2_size){

    if (max_spix>nspix_buffer/2){ return max_spix; }
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    // printf("nspix_buffer: %d\n",nspix_buffer);
    int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    // float alpha =  alpha;
    float a_0 = 1e4;
    // float a_0 = 1e6;
    float i_std = 0.018;
    i_std = sqrt(sigma2_app)*2;
    // i_std = sigma2_app;
    float b_0 = i_std * (a_0) ;
    // float b_0 = sigma2_app * (a_0) ;
    float alpha = exp(log_alpha);



    // float b_0;
    int* done_gpu;
    int* max_sp;
    int* nvalid;
    int nvalid_cpu;
    cudaMalloc((void **)&nvalid, sizeof(int));
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&done_gpu, sizeof(int)); 
    cudaMemset(nvalid, 0,sizeof(int));

    cudaMemset(sm_seg1, -1, npix*sizeof(int));
    cudaMemset(sm_seg2, -1, npix*sizeof(int));
    // cudaMemset(sm_seg1, 0, npix*sizeof(int));
    // cudaMemset(sm_seg2, 0, npix*sizeof(int));

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(img,seg,sp_params, sm_helper,
                                              nspix_buffer, nbatch, width,
                                              nftrs, npix, sm_pairs, nvalid);

    cudaMemcpy(&nvalid_cpu, nvalid, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("[split] nvalid: %d\n",nvalid_cpu);
    cudaMemset(nvalid, 0,sizeof(int));
    // printf("max_spix: %d\n",max_spix);
    // printf("max_sp: %d\n",max_sp);
    // exit(1);
    // printf("direction: %d\n",direction);


    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg1,sp_params,
                                                 nspix_buffer,
                                                 nbatch, width, height, direction,
                                                 seg, max_sp, max_spix);
    // printf("------------\n");

    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,sm_seg2,sp_params,
                                                 nspix_buffer,
                                                 nbatch, width,height, -direction,
                                                 seg, max_sp, max_spix);



    // -- compute sum of sm_seg2 --
    // sumIntArray(sm_seg1,height,width);
    // sumIntArray(sm_seg2,height,width);
    
    // idk what "split_sp" is doing here; init_sm clears the merge fields and
    // so the function returns immediately...
    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_spix);

    // printf("width,height,npix: %d,%d,%d\n",width,height,npix);

    int distance = 1;
    int done = 1;

    // printf("--> a.\n");
    while(done)
    {
      // // -- debug REMOVE ME --
      //   if (distance < 10){
      //     char buffer[50];
      //     sprintf(buffer, "sm_spix1_%d",distance);
      //     std::string fn = buffer;
      //     saveIntArray(sm_seg1, height, width, fn);
      //   }

        cudaMemset(done_gpu, 0, sizeof(int));
        // cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(\
                 sm_seg1,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("[a] distance: %d\n",distance);

        // // -- debug REMOVE ME --
        // if (distance > 5000){
        //   // saveIntArray(int* data, int H, int W, const std::string& filename) 
        //   // saveIntArray(sm_seg1,height,width,spir"sm_seg1.pth");
        //   char buffer[50];
        //   sprintf(buffer, "sm_spix1_%d",distance);
        //   std::string fn = buffer;
        //   saveIntArray(sm_seg1, height, width, fn);

          if (distance > 5005){
            exit(1); // check if mu_shape.x,y is set
          }
        // }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        // printf("a0.\n");
    }

    // printf("b.\n");
    done = 1;
    distance = 1;
    while(done)
    {
		cudaMemset(done_gpu, 0, sizeof(int));
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(\
                sm_seg2,seg,border,distance, done_gpu, npix, nbatch, width, height); 
        distance++;
        cudaMemcpy(&done, done_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("[b] distance: %d\n",distance);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        // printf("b0.\n");
    }
    // printf("c.\n");

    // updates the segmentation to the two regions; split either left/right or up/down.
    // printf("max_spix: %d\n",max_spix);
    calc_seg_split<<<BlockPerGrid,ThreadPerBlock>>>(sm_seg1,sm_seg2,
                                                    seg, npix, nbatch, max_spix);
    // std::string fname_split1_post = "split1_post";
    // write_tensor_to_file_v2(sm_seg1,height,width,fname_split1_post);

    // computes summaries stats for each split
    // printf("npix: %d\n",npix);
    sum_by_label<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1, sp_params,
                                                  sm_helper, npix, nbatch,
                                                  width,nftrs);

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
                                                        alpha, max_spix, max_sp);

    // -- do the split --
    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,sm_seg1,sm_pairs,
                                              sp_params, sm_helper, npix,
                                              nbatch, width, height, max_spix);


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
}






























/************************************************************



                       Merge Functions



*************************************************************/

// old name: calc_bn(int* seg
__global__ void calc_merge_stats_step0_p(int* sm_pairs,
                                       spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
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
    sm_helper[k].count_f = count_fk;
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
void calc_merge_stats_step1_p(spix_params* sp_params,spix_helper_sm* sm_helper,
                            const int nspix_buffer,float a_0, float b_0) {

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- read --
    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count_f);
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
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float alpha) {
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
    float count_f = __ldg(&sm_helper[k].count_f);
    
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


    sm_helper[k].hasting = log_nominator - log_denominator;

    // printf("[%2.2f,%2.2f]: %2.2f,%2.2f,%2.2f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,sm_helper[k].hasting,log_nominator,log_denominator);

    return;
}


__global__ void update_merge_flag_p(int* sm_pairs, spix_params* sp_params,
                                  spix_helper_sm* sm_helper, const int nspix_buffer,
                                  int* nmerges) {

	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = sm_pairs[2*k+1];
    // if(f<=0) return;
    if(f<0) return;
	if (sp_params[f].valid == 0) return;
    if((sm_helper[k].hasting ) > -2)
    {
      //printf("Want to merge k: %d, f: %d, splitmerge k %d, splitmerge  f %d, %d\n", k, f, sm_pairs[2*k], sm_pairs[2*f], sm_pairs[2*f+1] );
      int curr_max = atomicMax(&sm_pairs[2*f],k);
      if( curr_max == -1){
        atomicAdd(nmerges,1);
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
                                       spix_helper_sm* sm_helper,
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
                                       spix_helper_sm* sm_helper,
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
void update_split_flag_p(int* sm_pairs,
                       spix_params* sp_params,
                       spix_helper_sm* sm_helper,
                       const int nspix_buffer,
                       float alpha, int max_spix, int* max_sp ) {
  
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
    sm_helper[k].hasting = log_nominator - log_denominator;

    // -- check location --
    // printf("[%2.2f,%2.2f]: %2.3f, %2.3f, %2.3f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,
    //        sm_helper[k].hasting,log_nominator,log_denominator);
    // printf("[%2.2f,%2.2f]: %2.2f+%2.2f  = %2.2f\n",
    //        sp_params[k].mu_shape.x,sp_params[k].mu_shape.y,mu_s_x,mu_k_x,mu_f_x);
    // printf("hasting: %2.3f, %2.3f, %2.3f\n",
    //        sm_helper[k].hasting,log_nominator,log_denominator);

    // ".merge" is merely a bool variable; nothing about merging here. only splitting
    sm_helper[k].merge = (sm_helper[k].hasting > -2); // why "-2"?
    sm_helper[s].merge = (sm_helper[k].hasting > -2);

    if((sm_helper[k].merge)) // split step
      {

        s = atomicAdd(max_sp,1) +1; // ? can't multiple splits happen at one time? yes :D
        sm_pairs[2*k] = s;

        //atomicMax(max_sp,s);
        // sp_params[k].prior_count/=2;
        // sp_params[s].prior_count=  sp_params[k].prior_count; 
        float prior_count = max(sp_params[k].prior_count/2.0,8.0);
        sp_params[k].prior_count = prior_count;
        sp_params[s].prior_count = prior_count;
        // if (sp_params[k].valid == 0) return;

      }

}


















