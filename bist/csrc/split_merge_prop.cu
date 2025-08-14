
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

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

__host__
int run_split_p(const float* img, int* seg,
                int* shifted, bool* border,
                spix_params* sp_params,
                spix_helper* sp_helper,
                spix_helper_sm_v2* sm_helper,
                int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                float alpha_hastings, float split_alpha,
                float gamma,
                float sigma2_app, float sigma2_size, 
                int& count, int idx, int max_spix, 
                const int sp_size, const int npix,
                const int nbatch,
                const int width, const int height,
                const int nftrs, const int nspix_buffer,
                Logger* logger){

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
                                        gamma, sigma2_app, sigma2_size, logger);

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

                              // const int npix, const int nbatch,
                              // const int sp_size, const int width,
                              // const int height,
                              // const int nftrs, const int nspix_buffer,

__host__
void CudaCalcMergeCandidate_p(const float* img, int* seg,
                              bool* border, spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm_v2* sm_helper,
                              int* sm_pairs, float merge_alpha,
                              const int sp_size,
                              const int npix, const int nbatch,
                              const int width, const int height,
                              const int nftrs, const int nspix_buffer,
                              const int direction, float log_alpha,
                              float sigma2_app, float sigma2_size,
                              Logger* logger){

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

    merge_sp_p<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, sm_pairs,
                                                sp_params, sm_helper,
                                                npix, nbatch,
                                                width, height);  

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
                                      float split_alpha, float gamma,
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

    // computes summaries stats for each split
    sum_by_label_split_p<<<BlockPerGrid,ThreadPerBlock>>>(img, sm_seg1,
                                                          shifted, sp_params,
                                                          sm_helper, npix, nbatch,
                                                          height,width,nftrs,max_spix);


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
                                                          gamma, sp_size,
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
    sm_helper[k].ninvalid = 0;

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


}



__global__
void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
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
    // printf("x,y,C,C.prop: %d,%d,%d,%d\n",x,y,C,prop_C ? 1 : 0);
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

__global__
void sum_by_label_merge_p(const float* img,
                          const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs){
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

__global__
void sum_by_label_split_p(const float* img, const int* seg,
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

      }
    
    return;
    
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
                         float alpha, float split_alpha, float gamma,
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
    float hastings=log_nominator - log_denominator +split_alpha-gamma*(1.0-iperc_prop);
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























