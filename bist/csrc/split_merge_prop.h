


// -- cuda imports --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -- project --
#include "structs.h"

/************************************************************


                         API


*************************************************************/

__host__ int run_split_p(const float* img, int* seg,
                         int* shifted, bool* border,
                         spix_params* sp_params, spix_helper* sp_helper,
                         spix_helper_sm_v2* sm_helper,
                         int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                         float alpha_hastings, float split_alpha,
                         float iperc_coeff, float sigma2_app,
                         float sigma2_size,
                         int& count, int idx, int max_nspix,
                         const int sp_size, const int npix,
                         const int nbatch, const int width,
                         const int height, const int nftrs,
                         const int nspix_buffer, Logger* logger);

__host__ void run_merge_p(const float* img, int* seg, bool* border,
                        spix_params* sp_params, spix_helper* sp_helper,
                        spix_helper_sm_v2* sm_helper,
                        int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                          float merge_offset, float alpha_hastings,
                          float sigma2_app, float sigma2_size,
                        int& count, int idx, int max_nspix,
                        const int sp_size, const int npix,
                        const int nbatch, const int width,
                        const int height, const int nftrs,
                          const int nspix_buffer, Logger* logger);

__host__ void CudaCalcMergeCandidate_p(const float* img, int* seg,
                                     bool* border, spix_params* sp_params,
                                     spix_helper* sp_helper,
                                     spix_helper_sm_v2* sm_helper,int* sm_pairs,
                                       float merge_offset,
                                       const int npix, const int nbatch,
                                     const int sp_size,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha,
                                       float sigma2_app, float sigma2_size,Logger* logger);

__host__ int CudaCalcSplitCandidate_p(const float* img, int* seg,
                                      int* shifted, bool* border,
                                      spix_params* sp_params,
                                      spix_helper* sp_helper,
                                      spix_helper_sm_v2* sm_helper,
                                      int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                      const int sp_size,
                                      const int npix, const int nbatch, const int width,
                                      const int height, const int nftrs,
                                      const int nspix_buffer, int max_nspix,
                                      int direction,float alpha, float split_alpha,
                                      float iperc_coeff, float sigma2_app,
                                      float sigma2_size,Logger* logger);

__global__ void init_sm_p(const float* img,
                          const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int nspix_buffer, const int nbatch,
                          const int height, const int width,
                          const int nftrs, const int npix,
                          int* sm_pairs, int* nvalid);

/************************************************************


                       Merge Functions


*************************************************************/

__global__
void merge_marginal_likelihood_p(int* sm_pairs, spix_params* sp_params,
                               spix_helper_sm_v2* sm_helper,
                               const int sp_size,
                               const int npix, const int nbatch,
                               const int width, const int nspix_buffer,
                               float sigma2_app,float sigma2_size);
__global__
void merge_hastings_ratio_p(const float* img, int* sm_pairs,
                            spix_params* sp_params,
                            spix_helper* sp_helper,
                            spix_helper_sm_v2* sm_helper,
                            const int npix, const int nbatch, const int width,
                            const int nftrs, const int nspix_buffer,
                            float alpha_hasting_ratio, float merge_offset, int* nmerges);

/* __device__ double size_likelihood_p(int curr_count, int tgt_count, double sigma2); */
__device__ double size_likelihood_p(int curr_count, float tgt_count, double sigma2);
__device__ double size_likelihood_p_b(int curr_count, float tgt_count, double sigma2);

__global__  void calc_merge_candidate_p(int* seg, bool* border, int* sm_pairs,
                                        spix_params* sp_params,
                                        const int npix, const int nbatch,
                                        const int width, const int height,
                                        const int change);

__global__ void sum_by_label_merge_p(const float* img, const int* seg_gpu,
                                     spix_params* sp_params,
                                     spix_helper_sm_v2* sm_helper,
                                     const int npix, const int nbatch,
                                     const int width, const int nftrs);

__global__ void merge_likelihood_p(const float* img,int* sm_pairs,
                                 spix_params* sp_params,
                                 spix_helper* sp_helper,
                                 spix_helper_sm_v2* sm_helper,
                                 const int npix, const int nbatch,
                                 const int width, const int nftrs,
                                 const int nspix_buffer, float a_0, float b_0);


__global__ void remove_sp_p(int* sm_pairs, spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                            const int nspix_buffer, int* nmerges);

__global__ void merge_sp_p(int* seg, bool* border, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height);

/***********************************************************


                     Split Functions


************************************************************/

__global__ void init_split_p(const bool* border, int* seg_gpu,
                           spix_params* sp_params,
                           spix_helper_sm_v2* sm_helper,
                           const int nspix_buffer,
                           const int nbatch, const int width,
                           const int height, const int offset,
                           const int* seg, int* max_sp, int max_nspix);

__global__ void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int npix, const int nbatch,
                         const int width, const int height, int max_nspix);

__global__ void calc_split_candidate_p(int* dists, int* spix, bool* border,
                                     int distance, int* done_gpu, const int npix,
                                     const int nbatch, const int width, const int height);

__device__ double marginal_likelihood_shape_p(double3 sigma_est, double3 prior_sigma,
                                              float pr_count,int num_obs);

__device__ double compare_mu_pair(double3 mu0,double3 mu1);
__device__ double compare_mu_app_pair(double3 mu0,double3 mu1, int count0, int count1);
__device__ double compute_l2norm_mu_app_p(double3 sum_obs,float3 prior_mu,
                                          int _num_obs, double sigma2);

__global__ void calc_seg_split_p(int* sm_seg1, int* sm_seg2, int* seg,
                               const int npix, int nbatch, int max_nspix);

__global__ void sum_by_label_split_p(const float* img, const int* seg,
                                     int* shifted, spix_params* sp_params,
                                     spix_helper_sm_v2* sm_helper,
                                     const int npix, const int nbatch,
                                     const int height, const int width,
                                     const int nftrs, int max_nspix);

__global__
void sample_estimates_p(spix_params* sp_params,
                        spix_helper_sm_v2* sm_helper,
                        const int sp_size,
                        const int npix, const int nbatch,
                        const int width, const int nspix_buffer,
                        float sigma2_app, float sigma2_size,
                        int max_nspix);

__global__
void split_marginal_likelihood_p(spix_params* sp_params,
                                 spix_helper_sm_v2* sm_helper,
                                 const int sp_size,
                                 const int npix, const int nbatch,
                                 const int width, const int nspix_buffer,
                                 float sigma2_app, float sigma2_size,
                                 int max_nspix, int* count_rules);

__device__ double appearance_variance_p(double3 sum_obs,double3 sq_sum_obs,
                                      int _num_obs, double sigma2);
__device__ double marginal_likelihood_app_p(double3 sum_obs,double3 sq_sum_obs,
                                          int _num_obs, double sigma2);
__device__ double size_beta_likelihood_p(int _count, int _tgt_count,
                                       double alpha, const int _npix);

__device__ double compute_lprob_mu_app(double3 sum_obs,float3 prior_mu,
                                       int _num_obs, double sigma2);

__device__ double3 add_sigma_smoothing(double3 sigma_in, int count,float pc,int sp_size);
__device__ double3 compute_sigma_shape(longlong2 sum, longlong3 sq_sum,
                                       int count, float prior_count, int sp_size);
__device__ longlong2 get_sum_shape(longlong2 sum_s, longlong2 sum_k);
__device__ longlong3 get_sq_sum_shape(longlong3 sq_sum_s, longlong3 sq_sum_k);

__device__ double compute_lprob_sigma_shape(double3 sigma_est,
                                            double3 prior_sigma);
__device__ double wasserstein_p(double3 sigma_est,double3 sigma_prior);
__device__ double3 eigenvals_cov_pair(double3 icovA, double3 icovB,
                                      double detA, double detB);
__device__ double3 eigenvals_cov_p(double3 cov);

__global__
void split_likelihood_p(const float* img, int* sm_pairs,
                      spix_params* sp_params,
                      spix_helper* sp_helper,
                      spix_helper_sm_v2* sm_helper,
                      const int npix, const int nbatch,
                      const int width, const int nftrs,
                      const int nspix_buffer,
                      float a_0, float b_0, int max_nspix);

__global__
void split_hastings_ratio_p(const float* img, int* sm_pairs,
                          spix_params* sp_params,
                          spix_helper* sp_helper,
                          spix_helper_sm_v2* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs,
                          const int nspix_buffer,int sp_size,
                          float alpha_hasting_ratio,
                          int max_nspix, int* max_sp );

/************************************************************

                       Split Functions

*************************************************************/

__global__ void calc_split_stats_step0_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                       float b_0, int max_nspix);

__global__ void calc_split_stats_step1_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                       float a_0, float b_0, int max_nspix);

__global__
void update_split_flag_p(int* sm_pairs,
                         spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int nspix_buffer,
                         float alpha_hasting_ratio,
                         float split_alpha, float iperc_coeff,
                         int sp_size, int max_nspix, int* max_sp );


/************************************************************

                       Merge Functions

*************************************************************/

__global__ void calc_merge_stats_step0_p(int* sm_pairs, spix_params* sp_params,
                                       spix_helper* sp_helper, spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer, float b_0);
__global__ void calc_merge_stats_step1_p(spix_params* sp_params,spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,float a_0, float b_0);
__global__ void calc_merge_stats_step2_p(int* sm_pairs, spix_params* sp_params,
                                       spix_helper_sm_v2* sm_helper,
                                       const int nspix_buffer,
                                         float alpha_hasting_ratio, float merge_alpha);
__global__ void update_merge_flag_p(int* sm_pairs, spix_params* sp_params,
                                  spix_helper_sm_v2* sm_helper, const int nspix_buffer,
                                  int* nmerges);

