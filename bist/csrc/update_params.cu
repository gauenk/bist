
#include "update_params.h"
#include <math.h>
#define THREADS_PER_BLOCK 512


/***********************************************

           Compute Posterior Mode

************************************************/
__host__ void update_params(const float* img, const int* spix,
                            spix_params* sp_params,spix_helper* sp_helper,
                            float sigma2_app, const int npixels,
                            const int sp_size, const int nspix_buffer,
                            const int nbatch, const int width,
                            const int nftrs, bool prop_flag){

  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    int num_block1 = ceil( double(npixels) / double(THREADS_PER_BLOCK) ); 
	int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid1(num_block1,nbatch);
    dim3 BlockPerGrid2(num_block2,nbatch);
    clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                   nspix_buffer,nftrs);
    cudaMemset(sp_helper, 0, nbatch*nspix_buffer*sizeof(spix_helper));
    sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(img,spix,sp_params,sp_helper,
                                                   npixels,nbatch,width,nftrs,nspix_buffer);
    calc_simple_update<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                         sigma2_app, sp_size,
                                                         nspix_buffer, prop_flag);
}


// __host__ void update_params_log(const float* img, const int* spix,
//                                 spix_params* sp_params,spix_helper* sp_helper,
//                                 float sigma2_app, const int npixels,
//                                 const int sp_size, const int nspix_buffer,
//                                 const int nbatch, const int width,
//                                 const int nftrs, bool prop_flag, Logger* logger){

//   	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//     int num_block1 = ceil( double(npixels) / double(THREADS_PER_BLOCK) ); 
// 	int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
//     dim3 BlockPerGrid1(num_block1,nbatch);
//     dim3 BlockPerGrid2(num_block2,nbatch);
//     clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
//                                                    nspix_buffer,nftrs);
// 	cudaMemset(sp_helper, 0, nspix_buffer*sizeof(spix_helper));
//     sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(img,spix,sp_params,sp_helper,
//                                                    npixels,nbatch,width,nftrs);
//     calc_simple_update_log<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
//                                                          sigma2_app, sp_size,
//                                                          nspix_buffer, prop_flag,logger);
// }


__host__ void update_params_summ(const float* img, const int* spix,
                                 spix_params* sp_params,spix_helper* sp_helper,
                                 float sigma_app, const int npixels,
                                 const int nspix_buffer, const int nbatch,
                                 const int width, const int nftrs){

  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    int num_block1 = ceil( double(npixels) / double(THREADS_PER_BLOCK) ); 
	  int num_block2 = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid1(num_block1,nbatch);
    dim3 BlockPerGrid2(num_block2,nbatch);
    clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                   nspix_buffer,nftrs);
    cudaMemset(sp_helper, 0, nspix_buffer*sizeof(spix_helper));
    sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(img,spix,sp_params,sp_helper,
                                                   npixels,nbatch,width,nftrs,nspix_buffer);
    calc_summ_stats<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_helper,
                                                      sigma_app,nspix_buffer); 
}

__global__
void clear_fields(spix_params* sp_params,
                  spix_helper* sp_helper,
                  const int nsuperpixel_buffer,
                  const int nftrs){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	k = k + nsuperpixel_buffer*blockIdx.y; // batch dimension
	// if (sp_params[k].valid == 0) return;

	sp_params[k].count = 0;
	float3 mu_app;
	mu_app.x = 0;
	mu_app.y = 0;
	mu_app.z = 0;
	sp_params[k].mu_app = mu_app;

    // float3 sigma_app;
    // sigma_app.x = 0;
    // sigma_app.y = 0;
    // sigma_app.z = 0;
	// sp_params[k].sigma_app = sigma_app;

	double2 mu_shape;
	mu_shape.x = 0;
	mu_shape.y = 0;
	sp_params[k].mu_shape = mu_shape;

	double3 sigma_shape;
	sigma_shape.x = 0;
	sigma_shape.y = 0;
	sigma_shape.z = 0;
	sp_params[k].sigma_shape = sigma_shape;

}
__global__
void sum_by_label(const float* img, const int* spix,
                  spix_params* sp_params, spix_helper* sp_helper,
                  const int npixels, const int nbatch,
                  const int width, const int nftrs, const int nspix_buffer) {

    // --> getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t>=npixels) return;
    // --> get the label
    int batch_ix = blockIdx.y;
    t = t + npixels*batch_ix;

    int k = spix[t];
    if (k < 0){ return; } // invalid label
    int label_offset = nspix_buffer * batch_ix;
    k = k + label_offset;
    // if (sp_params[k].valid != 1){
    //   printf("invalid, living spix id %d\n",k);
    // }
    assert(sp_params[k].valid==1);

    const float* _img = img+3*t;
    float3 pix;
    pix.x = _img[0];
    pix.y = _img[1];
    pix.z = _img[2];
    // float3 pix = *(float3*)(img+3*t);
	atomicAdd(&sp_params[k].count, 1);
	atomicAdd(&sp_helper[k].sum_app.x, pix.x);
	atomicAdd(&sp_helper[k].sum_app.y, pix.y);
	atomicAdd(&sp_helper[k].sum_app.z, pix.z);
	// atomicAdd(&sp_helper[k].sq_sum_app.x, pix.x*pix.x);
	// atomicAdd(&sp_helper[k].sq_sum_app.y, pix.y*pix.y);
	// atomicAdd(&sp_helper[k].sq_sum_app.z, pix.z*pix.z);

	int x = t % width;
	int y = t / width; 
	atomicAdd((unsigned long long *)&sp_helper[k].sum_shape.x, x);
	atomicAdd((unsigned long long *)&sp_helper[k].sum_shape.y, y);
    atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.x, x*x);
	atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.y, x*y);
	atomicAdd((unsigned long long *)&sp_helper[k].sq_sum_shape.z, y*y);
	
}


/***********************************************************

          Summary Statistics for
          -> Normal-Inverse-Gamma for Appearance (_app)
          -> Normal-Inverse-Wishart for Shape (_shape)

************************************************************/

__global__
void calc_summ_stats(spix_params*  sp_params,spix_helper* sp_helper,
                     float sigma_app, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
  k = k + nsuperpixel_buffer*blockIdx.y; // batch dimension
	if (sp_params[k].valid == 0) return;
    
    // -- read curr --
	int count_int = sp_params[k].count;
	float a_prior = sp_params[k].prior_count;
	float prior_sigma_s_2 = a_prior * a_prior;
	double count = count_int * 1.0;
    double2 mu_shape;
    float3 mu_app;
    double3 sigma_shape;

    // --  sample means --
	// if (count_int<=0){ return; }
	if (count_int<=0){
      sp_params[k].valid = 0; // invalidate empty spix?
      return;
    }

    // -- appearance --
    mu_app.x = sp_helper[k].sum_app.x / count;
    mu_app.y = sp_helper[k].sum_app.y / count;
    mu_app.z = sp_helper[k].sum_app.z / count;
    sp_params[k].mu_app = mu_app;

    // -- view --
    // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
    // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
    // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

    // -- shape --
    mu_shape.x = sp_helper[k].sum_shape.x / count;
    mu_shape.y = sp_helper[k].sum_shape.y / count;
    sp_params[k].mu_shape = mu_shape;
    // sp_params[k].mu_shape.x = mu_shape.x;
    // sp_params[k].mu_shape.y = mu_shape.y;

    // -- sample covariance [NOT inverse] for shape --
    sigma_shape.x = sp_helper[k].sq_sum_shape.x/count - mu_shape.x*mu_shape.x;
    sigma_shape.y = sp_helper[k].sq_sum_shape.y/count - mu_shape.x*mu_shape.y;
    sigma_shape.z = sp_helper[k].sq_sum_shape.z/count - mu_shape.y*mu_shape.y;

    // -- correct sample cov if not invertable --
    double det = sigma_shape.x*sigma_shape.z - sigma_shape.y*sigma_shape.y;
    if (det <= 0){
      sigma_shape.x = sigma_shape.x + 0.00001;
      sigma_shape.z = sigma_shape.z + 0.00001;
      det = sigma_shape.x * sigma_shape.z - sigma_shape.y * sigma_shape.y;
      if (det<=0){ det = 0.00001; } // safety hack
    }

    sp_params[k].sigma_shape.x = sigma_shape.x;
    sp_params[k].sigma_shape.y = sigma_shape.y;
    sp_params[k].sigma_shape.z = sigma_shape.z;
    sp_params[k].logdet_sigma_shape = log(det);

}


__device__ double3 _add_sigma_smoothing(double3 in_sigma, int count,
                                       float pc, int sp_size) {
  // -- sample cov --
  int nf = 1;
  pc = pc*0.25;
  double total_count = 1.*(count + pc);
  double3 sigma;
  sigma.x = (pc*sp_size + count*in_sigma.x)/(total_count + 3.0);
  sigma.y = (count*in_sigma.y) / (total_count + 3.0);
  sigma.z = (pc*sp_size + count*in_sigma.z)/(total_count + 3.0);
  return sigma;
  // return in_sigma;
}


__global__
void calc_simple_update(spix_params*  sp_params,spix_helper* sp_helper,
                        float sigma_app, const int sp_size,
                        const int nsuperpixel_buffer, bool prop_flag) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
	k = k + nsuperpixel_buffer*blockIdx.y;
    
    // -- read curr --
	int count_int = sp_params[k].count;
	float pc = sp_params[k].prior_count; 
	float prior_sigma_s_2 = pc * pc;
	// float prior_sigma_s_2 = pc * sp_size;
	double count = count_int * 1.0;
    double2 mu_shape;
    float3 mu_app;
    double3 sigma_shape;
	// double total_count = (double) count_int + pc;
	// double total_count = (double) count_int + pc*50;
	double total_count = (double) count_int + pc*50;

    // --  sample means --
	// if (count_int<=0){ return; }
	if (count_int<=0){
      sp_params[k].valid = 0; // invalidate empty spix?
      return;
    }

    // -- get prior --
    double3 prior_sigma_shape;
    // double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
    // double pr_det = prior_sigma_shape.x*prior_sigma_shape.z - \
    //   prior_sigma_shape.y*prior_sigma_shape.y;
    // // printf("[a:%d] sxx,syx,syy: %lf %lf %lf %lf\n",
    // //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pr_det);
    // double tmp = prior_sigma_shape.x;
    // prior_sigma_shape.x = prior_sigma_shape.z/pr_det;
    // prior_sigma_shape.y = prior_sigma_shape.y/pr_det;
    // prior_sigma_shape.z = tmp/pr_det;
    // prior_icov_eig.
    // printf("[b:%d] sxx,syx,syy: %lf %lf %lf %f\n",
    //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pc);
    // prior_sigma_shape.x = pc * sp_size; // not me
    // prior_sigma_shape.y = 0;
    // prior_sigma_shape.z = pc * sp_size; // not me

    // -- [deploy] prior shape --
    bool prop = sp_params[k].prop;
    double3 prior_icov;
    double pr_det,pr_det_raw;
    double3 _prior_icov = sp_params[k].sample_sigma_shape;
    // if ((prop) and false){
    if ((prop) and prop_flag){
      prior_icov = _add_sigma_smoothing(_prior_icov,pc,pc,sp_size);
      pr_det = prior_icov.x * prior_icov.z  - \
        prior_icov.y * prior_icov.y;
      if (pr_det <= 0){
        printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",prior_icov.x,prior_icov.y,prior_icov.z);
        assert(pr_det>0);
      }
      pr_det_raw = pr_det;
      pr_det = sqrt(pr_det);
      // double pc_sqrt = sqrt(pc);
      prior_sigma_shape.x = pc/pr_det * prior_icov.x;
      prior_sigma_shape.y = pc/pr_det * prior_icov.y;
      prior_sigma_shape.z = pc/pr_det * prior_icov.z;
    }else{
      prior_icov = sp_params[k].prior_icov;
      pr_det = prior_icov.x * prior_icov.z  - \
        prior_icov.y * prior_icov.y;
      if (pr_det <= 0){
        printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",prior_icov.x,prior_icov.y,prior_icov.z);
        assert(pr_det>0);
      }
      pr_det_raw = pr_det;
      pr_det = sqrt(pr_det);
      // double pc_sqrt = sqrt(pc);
      prior_sigma_shape.x = pc/pr_det * prior_icov.x;
      prior_sigma_shape.y = pc/pr_det * prior_icov.y;
      prior_sigma_shape.z = pc/pr_det * prior_icov.z;
    }
    // if (k == 100){
    //   printf("[update_params]: %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf\n",
    //          // pc,pr_det,prior_icov.x,prior_icov.y,prior_icov.z);
    //          pc,pr_det,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);
    // }


    // -- [dev] prior shape --
    // prior_sigma_shape.x = pc;// * pc;
    // prior_sigma_shape.y = 0;
    // prior_sigma_shape.z = pc;// * pc;

    // -- appearance --
    mu_app.x = sp_helper[k].sum_app.x / count;
    mu_app.y = sp_helper[k].sum_app.y / count;
    mu_app.z = sp_helper[k].sum_app.z / count;
    sp_params[k].mu_app = mu_app;

    // mu_app.x = 0;
    // mu_app.y = 0;
    // mu_app.z = 0;
    // if ((abs(mu_app.x)>10) || (abs(mu_app.y)>10) || (abs(mu_app.z)>10)){
    //   printf("[updated_params.cu] [%d] %2.3f, %2.3f, %2.3f\n",k,mu_app.x,mu_app.y,mu_app.z);
    // }
    // sp_params[k].mu_app.x = mu_app.x;
    // sp_params[k].mu_app.y = mu_app.y;
    // sp_params[k].mu_app.z = mu_app.z;
    // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
    // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
    // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

    // -- shape --
    mu_shape.x = sp_helper[k].sum_shape.x / count;
    mu_shape.y = sp_helper[k].sum_shape.y / count;
    sp_params[k].mu_shape = mu_shape;
    // sp_params[k].mu_shape.x = mu_shape.x;
    // sp_params[k].mu_shape.y = mu_shape.y;

    // -- sample covariance --
    sigma_shape.x = sp_helper[k].sq_sum_shape.x - count*mu_shape.x*mu_shape.x;
    sigma_shape.y = sp_helper[k].sq_sum_shape.y - count*mu_shape.x*mu_shape.y;
    sigma_shape.z = sp_helper[k].sq_sum_shape.z - count*mu_shape.y*mu_shape.y;

    // -- inverse --
    // sigma_shape.x = (prior_sigma_s_2 + sigma_shape.x) / (total_count - 3.0);
    // sigma_shape.y = sigma_shape.y / (total_count - 3);
    // sigma_shape.z = (prior_sigma_s_2 + sigma_shape.z) / (total_count - 3.0);
    sigma_shape.x = (pc*prior_sigma_shape.x + sigma_shape.x) / (total_count + 3.0);
    sigma_shape.y = (pc*prior_sigma_shape.y + sigma_shape.y) / (total_count + 3.0);
    sigma_shape.z = (pc*prior_sigma_shape.z + sigma_shape.z) / (total_count + 3.0);

    // -- correct sample cov if not invertable --
    double det = sigma_shape.x*sigma_shape.z - sigma_shape.y*sigma_shape.y;
    if (det <= 0){
      sigma_shape.x = sigma_shape.x + 0.00001;
      sigma_shape.z = sigma_shape.z + 0.00001;
      det = sigma_shape.x * sigma_shape.z - sigma_shape.y * sigma_shape.y;
      if (det<=0){ det = 0.00001; } // safety hack
    }

    bool any_nan = isnan(sigma_shape.x) and isnan(sigma_shape.y) and isnan(sigma_shape.z);
    if ((any_nan)){
      printf("[%d|%d] %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf\n",
             k,prop?1:0,pc,pr_det,pr_det_raw,
             prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,
             prior_icov.x,prior_icov.y,prior_icov.z,
             _prior_icov.x,_prior_icov.y,_prior_icov.z);

    }
    assert(not(any_nan));
    // assert(not(isnan(sigma_shape.x)));
    // assert(not(isnan(sigma_shape.y)));
    // assert(not(isnan(sigma_shape.z)));

    sp_params[k].sigma_shape.x = sigma_shape.z/det;
    sp_params[k].sigma_shape.y = -sigma_shape.y/det;
    sp_params[k].sigma_shape.z = sigma_shape.x/det;
    sp_params[k].logdet_sigma_shape = log(det);

}


// __global__
// void calc_simple_update_log(spix_params*  sp_params,spix_helper* sp_helper,
//                         float sigma_app, const int sp_size,
//                             const int nsuperpixel_buffer, bool prop_flag, Logger* logger) {

//     // -- update thread --
// 	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
// 	if (k>=nsuperpixel_buffer) return;
// 	if (sp_params[k].valid == 0) return;
    
//     // -- read curr --
// 	int count_int = sp_params[k].count;
// 	float pc = sp_params[k].prior_count; 
// 	float prior_sigma_s_2 = pc * pc;
// 	// float prior_sigma_s_2 = pc * sp_size;
// 	double count = count_int * 1.0;
//     double2 mu_shape;
//     float3 mu_app;
//     double3 sigma_shape;
// 	// double total_count = (double) count_int + pc;
// 	// double total_count = (double) count_int + pc*50;
// 	double total_count = (double) count_int + pc*50;

//     // --  sample means --
// 	// if (count_int<=0){ return; }
// 	if (count_int<=0){
//       sp_params[k].valid = 0; // invalidate empty spix?
//       return;
//     }

//     // -- get prior --
//     double3 prior_sigma_shape;
//     // double3 prior_sigma_shape = sp_params[k].prior_sigma_shape;
//     // double pr_det = prior_sigma_shape.x*prior_sigma_shape.z - \
//     //   prior_sigma_shape.y*prior_sigma_shape.y;
//     // // printf("[a:%d] sxx,syx,syy: %lf %lf %lf %lf\n",
//     // //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pr_det);
//     // double tmp = prior_sigma_shape.x;
//     // prior_sigma_shape.x = prior_sigma_shape.z/pr_det;
//     // prior_sigma_shape.y = prior_sigma_shape.y/pr_det;
//     // prior_sigma_shape.z = tmp/pr_det;
//     // prior_icov_eig.
//     // printf("[b:%d] sxx,syx,syy: %lf %lf %lf %f\n",
//     //        k,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,pc);
//     // prior_sigma_shape.x = pc * sp_size; // not me
//     // prior_sigma_shape.y = 0;
//     // prior_sigma_shape.z = pc * sp_size; // not me

//     // -- [deploy] prior shape --
//     bool prop = sp_params[k].prop;
//     double3 prior_icov;
//     double pr_det,pr_det_raw;
//     double3 _prior_icov = sp_params[k].sample_sigma_shape;
//     // if ((prop) and false){
//     if ((prop) and prop_flag){
//       prior_icov = _add_sigma_smoothing(_prior_icov,pc,pc,sp_size);
//       pr_det = prior_icov.x * prior_icov.z  - \
//         prior_icov.y * prior_icov.y;
//       if (pr_det <= 0){
//         printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",prior_icov.x,prior_icov.y,prior_icov.z);
//         assert(pr_det>0);
//       }
//       pr_det_raw = pr_det;
//       pr_det = sqrt(pr_det);
//       // double pc_sqrt = sqrt(pc);
//       prior_sigma_shape.x = pc/pr_det * prior_icov.x;
//       prior_sigma_shape.y = pc/pr_det * prior_icov.y;
//       prior_sigma_shape.z = pc/pr_det * prior_icov.z;
//     }else{
//       prior_icov = sp_params[k].prior_icov;
//       pr_det = prior_icov.x * prior_icov.z  - \
//         prior_icov.y * prior_icov.y;
//       if (pr_det <= 0){
//         printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",prior_icov.x,prior_icov.y,prior_icov.z);
//         assert(pr_det>0);
//       }
//       pr_det_raw = pr_det;
//       pr_det = sqrt(pr_det);
//       // double pc_sqrt = sqrt(pc);
//       prior_sigma_shape.x = pc/pr_det * prior_icov.x;
//       prior_sigma_shape.y = pc/pr_det * prior_icov.y;
//       prior_sigma_shape.z = pc/pr_det * prior_icov.z;
//     }
//     // if (k == 100){
//     //   printf("[update_params]: %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf\n",
//     //          // pc,pr_det,prior_icov.x,prior_icov.y,prior_icov.z);
//     //          pc,pr_det,prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z);
//     // }


//     // -- [dev] prior shape --
//     // prior_sigma_shape.x = pc;// * pc;
//     // prior_sigma_shape.y = 0;
//     // prior_sigma_shape.z = pc;// * pc;

//     // -- appearance --
//     mu_app.x = sp_helper[k].sum_app.x / count;
//     mu_app.y = sp_helper[k].sum_app.y / count;
//     mu_app.z = sp_helper[k].sum_app.z / count;
//     sp_params[k].mu_app = mu_app;

//     // mu_app.x = 0;
//     // mu_app.y = 0;
//     // mu_app.z = 0;
//     // if ((abs(mu_app.x)>10) || (abs(mu_app.y)>10) || (abs(mu_app.z)>10)){
//     //   printf("[updated_params.cu] [%d] %2.3f, %2.3f, %2.3f\n",k,mu_app.x,mu_app.y,mu_app.z);
//     // }
//     // sp_params[k].mu_app.x = mu_app.x;
//     // sp_params[k].mu_app.y = mu_app.y;
//     // sp_params[k].mu_app.z = mu_app.z;
//     // sp_params[k].sigma_app.x = sp_helper[k].sq_sum_app.x/count - mu_app.x*mu_app.x;
//     // sp_params[k].sigma_app.y = sp_helper[k].sq_sum_app.y/count - mu_app.y*mu_app.y;
//     // sp_params[k].sigma_app.z = sp_helper[k].sq_sum_app.z/count - mu_app.z*mu_app.z;

//     // -- shape --
//     mu_shape.x = sp_helper[k].sum_shape.x / count;
//     mu_shape.y = sp_helper[k].sum_shape.y / count;
//     sp_params[k].mu_shape = mu_shape;
//     // sp_params[k].mu_shape.x = mu_shape.x;
//     // sp_params[k].mu_shape.y = mu_shape.y;

//     // -- sample covariance --
//     sigma_shape.x = sp_helper[k].sq_sum_shape.x - count*mu_shape.x*mu_shape.x;
//     sigma_shape.y = sp_helper[k].sq_sum_shape.y - count*mu_shape.x*mu_shape.y;
//     sigma_shape.z = sp_helper[k].sq_sum_shape.z - count*mu_shape.y*mu_shape.y;

//     // -- inverse --
//     // sigma_shape.x = (prior_sigma_s_2 + sigma_shape.x) / (total_count - 3.0);
//     // sigma_shape.y = sigma_shape.y / (total_count - 3);
//     // sigma_shape.z = (prior_sigma_s_2 + sigma_shape.z) / (total_count - 3.0);
//     sigma_shape.x = (pc*prior_sigma_shape.x + sigma_shape.x) / (total_count + 3.0);
//     sigma_shape.y = (pc*prior_sigma_shape.y + sigma_shape.y) / (total_count + 3.0);
//     sigma_shape.z = (pc*prior_sigma_shape.z + sigma_shape.z) / (total_count + 3.0);

//     // -- correct sample cov if not invertable --
//     double det = sigma_shape.x*sigma_shape.z - sigma_shape.y*sigma_shape.y;
//     if (det <= 0){
//       sigma_shape.x = sigma_shape.x + 0.00001;
//       sigma_shape.z = sigma_shape.z + 0.00001;
//       det = sigma_shape.x * sigma_shape.z - sigma_shape.y * sigma_shape.y;
//       if (det<=0){ det = 0.00001; } // safety hack
//     }

//     bool any_nan = isnan(sigma_shape.x) and isnan(sigma_shape.y) and isnan(sigma_shape.z);
//     if ((any_nan)){
//       printf("[%d|%d] %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf | %2.3lf %2.3lf %2.3lf\n",
//              k,prop?1:0,pc,pr_det,pr_det_raw,
//              prior_sigma_shape.x,prior_sigma_shape.y,prior_sigma_shape.z,
//              prior_icov.x,prior_icov.y,prior_icov.z,
//              _prior_icov.x,_prior_icov.y,_prior_icov.z);

//     }
//     assert(not(any_nan));
//     // assert(not(isnan(sigma_shape.x)));
//     // assert(not(isnan(sigma_shape.y)));
//     // assert(not(isnan(sigma_shape.z)));

//     sp_params[k].sigma_shape.x = sigma_shape.z/det;
//     sp_params[k].sigma_shape.y = -sigma_shape.y/det;
//     sp_params[k].sigma_shape.z = sigma_shape.x/det;
//     sp_params[k].logdet_sigma_shape = log(det);

// }


__host__
void store_sample_sigma_shape(spix_params* sp_params,spix_helper* sp_helper,
                              const int sp_size, const int nspix_buffer){
  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
	int num_block = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid(num_block,1);
    store_sample_sigma_shape_k<<<BlockPerGrid,ThreadPerBlock>>>(sp_params,sp_helper,
                                                                sp_size,nspix_buffer);
}


__global__
void store_sample_sigma_shape_k(spix_params*  sp_params,spix_helper* sp_helper,
                                float sigma_app, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

    // -- unpack --
	int count_int = sp_params[k].count;
	double count = count_int * 1.0;
    
    // -- shape --
    double2 mu_shape;
    mu_shape.x = sp_helper[k].sum_shape.x / count;
    mu_shape.y = sp_helper[k].sum_shape.y / count;
    sp_params[k].mu_shape = mu_shape;

    // -- sample covariance --
    double3 sigma_shape;
    if (count > 0){
      sigma_shape.x = sp_helper[k].sq_sum_shape.x/count - mu_shape.x*mu_shape.x;
      sigma_shape.y = sp_helper[k].sq_sum_shape.y/count - mu_shape.x*mu_shape.y;
      sigma_shape.z = sp_helper[k].sq_sum_shape.z/count - mu_shape.y*mu_shape.y;
    }else{
      sigma_shape = sp_params[k].prior_sigma_shape;
    }

    // sigma_shape.x = 1.;
    // sigma_shape.y = 0.;
    // sigma_shape.z = 1.;

    // -- nan check  --
    bool any_nan = isnan(sigma_shape.x) and isnan(sigma_shape.y) and isnan(sigma_shape.z);
    if ((any_nan)){
      printf("[stores_sample:%d] %2.3lf %2.3lf %2.3lf\n",
             k,sigma_shape.x,sigma_shape.y,sigma_shape.z);
    }
    assert(not(any_nan));

    sp_params[k].sample_sigma_shape = sigma_shape;
}