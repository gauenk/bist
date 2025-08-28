
#include "update_params_3d.h"
#include <math.h>
#define THREADS_PER_BLOCK 512


/***********************************************

           Compute Posterior Mode

************************************************/


__host__ void update_params(spix_params* aos_params, spix_helper* sp_helper, PointCloudData& data, SuperpixelParams3d& soa_params, SpixMetaData& args, Logger* logger){
 

    // -- launch parameters --
    int nspix_buffer = soa_params.nspix_sum * args.nspix_buffer_mult;

    int NumThreads = THREADS_PER_BLOCK;
    int vertex_nblocks = ceil( double(data.V) / double(THREADS_PER_BLOCK) ); 
    dim3 VertexBlocks(vertex_nblocks);

	int spix_nblocks = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 SpixBlocks(spix_nblocks);

    // -- clear all but the "valid" bool --
    clear_fields<<<SpixBlocks,NumThreads>>>(aos_params,sp_helper,nspix_buffer);
    cudaMemset(sp_helper, 0, nspix_buffer*sizeof(spix_helper));

    // -- accumulate via sum --
    sum_by_label<<<VertexBlocks,NumThreads>>>(data.ftrs,data.pos,data.ptr,data.bids,
                                              soa_params.spix_ptr(),soa_params.csum_nspix_ptr(),
                                              aos_params,sp_helper,data.V,nspix_buffer);

    // -- compute sample stats --
    update_params_kernel<<<SpixBlocks,NumThreads>>>(aos_params,sp_helper,args.sigma2_app, args.sp_size, nspix_buffer);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


__global__
void clear_fields(spix_params* sp_params, spix_helper* sp_helper, const int nspix_total){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_total) return;

	sp_params[k].count = 0;
    sp_params[k].logdet_sigma_shape = 0;

	float3 mu_app;
	mu_app.x = 0;
	mu_app.y = 0;
	mu_app.z = 0;
	sp_params[k].mu_app = mu_app;

	double3 mu_pos;
	mu_pos.x = 0;
	mu_pos.y = 0;
	mu_pos.z = 0;
	sp_params[k].mu_pos = mu_pos;

	double3 var_pos;
	var_pos.x = 0;
	var_pos.y = 0;
	var_pos.z = 0;
	sp_params[k].var_pos = var_pos;

	double3 cov_pos;
	cov_pos.x = 0;
	cov_pos.y = 0;
	cov_pos.z = 0;
	sp_params[k].cov_pos = cov_pos;
}

__global__
void sum_by_label(const float3* ftrs, const float3* pos,
                  const int* vptr, const uint8_t* vbids,
                  const uint32_t* spix, const uint32_t* csum_nspix,
                  spix_params* sp_params, spix_helper* sp_helper,
                  const int V_total, const int nspix_buffer) {

    // -- get vertex --
    int vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex>=V_total) return;
    int bx = vbids[vertex];
    int V = vptr[bx+1] - vptr[bx];
    uint32_t spix_offset = csum_nspix[bx];

    // -- read superpixel label and some checks --
    uint32_t spix_id = spix[vertex];
    if (spix_id < 0){
        printf("invalid superpixel id.\n");
    }
    assert(spix_id >= 0);

    int spix_index = spix_id + spix_offset;
    if (sp_params[spix_index].valid != 1){
      printf("invalid but living spix[(%d,%d)]: %d %d\n",bx,vertex,spix_id,spix_index);
    }
    assert(sp_params[spix_index].valid==1);

    // -- unpack data --
    const float3 v_ftr = ftrs[vertex];
    const float3 v_pos = pos[vertex];

    // -- write features --
	atomicAdd(&sp_params[spix_index].count, 1);
    atomicAdd(&sp_helper[spix_index].sum_app.x, v_ftr.x);
    atomicAdd(&sp_helper[spix_index].sum_app.y, v_ftr.y);
    atomicAdd(&sp_helper[spix_index].sum_app.z, v_ftr.z);

    // -- write positions --
	atomicAdd(&sp_helper[spix_index].sum_pos.x, v_pos.x);
	atomicAdd(&sp_helper[spix_index].sum_pos.y, v_pos.y);
	atomicAdd(&sp_helper[spix_index].sum_pos.z, v_pos.z);

    atomicAdd(&sp_helper[spix_index].sq_sum_self_pos.x, v_pos.x*v_pos.x);
	atomicAdd(&sp_helper[spix_index].sq_sum_self_pos.y, v_pos.y*v_pos.y);
	atomicAdd(&sp_helper[spix_index].sq_sum_self_pos.z, v_pos.z*v_pos.z);

    atomicAdd(&sp_helper[spix_index].sq_sum_pairs_pos.x, v_pos.x*v_pos.y);
	atomicAdd(&sp_helper[spix_index].sq_sum_pairs_pos.y, v_pos.x*v_pos.z);
	atomicAdd(&sp_helper[spix_index].sq_sum_pairs_pos.z, v_pos.y*v_pos.z);
	
}

__global__
void update_params_kernel(spix_params* sp_params, spix_helper* sp_helper,
                          float sigma_app, const int sp_size, const int nsuperpixel_buffer) {

    // -- update thread --
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
//   // -- read curr --
// 	int count_int = sp_params[k].count;
// 	float pc = sp_params[k].prior_count; 
// 	float prior_sigma_s_2 = pc * pc;
// 	// float prior_sigma_s_2 = pc * sp_size;
// 	double count = count_int * 1.0;
//   double2 mu_shape;
//   float3 mu_app;
//   double3 sigma_shape;
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
//         printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",k,prior_icov.x,prior_icov.y,prior_icov.z);
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
//         printf("[%d]: (%2.3lf,%2.3lf,%2.3lf)\n",k,prior_icov.x,prior_icov.y,prior_icov.z);
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

}

