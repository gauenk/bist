


// -- cuda imports --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -- project --
#include "structs.h"

/************************************************************


                         API


*************************************************************/

__host__
int run_split_p(const float* img, int* seg,
                int* shifted, bool* border,
                spix_params* sp_params,
                spix_helper* sp_helper,
                spix_helper_sm_v2* sm_helper,
                int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                float alpha_hastings, float split_alpha,
                float iperc_coeff,
                float sigma2_app, float sigma2_size,
                int& count, int idx, int max_spix,
                const int sp_size, const int npix,
                const int nbatch,
                const int width, const int height,
                const int nftrs, const int nspix_buffer,
                Logger* logger);

/* __host__ int run_split_p(const float* img, int* seg, */
/*                          int* shifted, bool* border, */
/*                          spix_params* sp_params, spix_helper* sp_helper, */
/*                          spix_helper_sm_v2* sm_helper, */
/*                          int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                          float alpha_hastings, float split_alpha, */
/*                          float iperc_coeff, float sigma2_app, */
/*                          float sigma2_size, */
/*                          int& count, int idx, int max_nspix, */
/*                          const int sp_size, const int npix, */
/*                          const int nbatch, const int width, */
/*                          const int height, const int nftrs, */
/*                          const int nspix_buffer, Logger* logger); */


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
                 const int nftrs, const int nspix_buffer, Logger* logger);

/* __host__ void run_merge_p(const float* img, int* seg, bool* border, */
/*                         spix_params* sp_params, spix_helper* sp_helper, */
/*                         spix_helper_sm_v2* sm_helper, */
/*                         int* sm_seg1 ,int* sm_seg2, int* sm_pairs, */
/*                           float merge_offset, float alpha_hastings, */
/*                           float sigma2_app, float sigma2_size, */
/*                         int& count, int idx, int max_nspix, */
/*                         const int sp_size, const int npix, */
/*                         const int nbatch, const int width, */
/*                         const int height, const int nftrs, */
/*                           const int nspix_buffer, Logger* logger); */

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
                              Logger* logger);

/* __host__ */
/* void CudaCalcMergeCandidate_p(const float* img, int* seg, */
/*                               bool* border, spix_params* sp_params, */
/*                               spix_helper* sp_helper, */
/*                               spix_helper_sm_v2* sm_helper, */
/*                               int* sm_pairs, */
/*                               float merge_offset, */
/*                               const int npix, const int nbatch, */
/*                               const int width, const int height, */
/*                               const int nftrs, const int sp_size, */
/*                               const int nspix_buffer, */
/*                               const int direction, float alpha, */
/*                               float sigma2_app, float sigma2_size, */
/*                               Logger* logger); */

/* __host__ */
/* int CudaCalcSplitCandidate_p(const float* img, int* seg, */
/*                              int* shifted, bool* border, */
/*                              spix_params* sp_params, */
/*                              spix_helper* sp_helper, */
/*                              spix_helper_sm_v2* sm_helper, */
/*                              int* sm_seg1, int* sm_seg2, int* sm_pairs, */
/*                              const int sp_size, */
/*                              const int npix, const int nbatch, */
/*                              const int width, const int height, */
/*                              const int nftrs, const int nspix_buffer, */
/*                              int max_nspix, int direction,float alpha, */
/*                              float split_alpha, float iperc_coeff, */
/*                              float sigma2_app, float sigma2_size, */
/*                              Logger* logger); */

__host__
int CudaCalcSplitCandidate_p(const float* img, int* seg,
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
                             float sigma2_app, float sigma2_size, Logger* logger);


__global__
void init_sm_p(const float* img, const int* seg_gpu,
               spix_params* sp_params,
               spix_helper_sm_v2* sm_helper,
               const int nspix_buffer, const int nbatch,
               const int height, const int width,
               const int nftrs, const int npix,
               int* sm_pairs, int* nvalid);


/***********************************************************


                     Split Proposal Functions


************************************************************/

__global__
void init_split_p(const bool* border, int* seg_gpu,
                  spix_params* sp_params,
                  spix_helper_sm_v2* sm_helper,
                  const int nspix_buffer,
                  const int nbatch, const int width,
                  const int height, const int offset,
                  const int* seg, int* max_sp, int max_nspix);

__global__
void split_sp_p(int* seg, int* sm_seg1, int* sm_pairs,
                spix_params* sp_params,
                spix_helper_sm_v2* sm_helper,
                const int npix, const int nbatch,
                const int width, const int height, int max_nspix);

__global__
void calc_split_candidate_p(int* dists, int* spix,
                            bool* border, int distance,
                            int* done_gpu, const int npix,
                            const int nbatch,
                            const int width, const int height);

__global__
void calc_seg_split_p(int* sm_seg1, int* sm_seg2,
                      int* seg, const int npix,
                      int nbatch, int max_nspix);

__global__
void sum_by_label_split_p(const float* img, const int* seg,
                          int* shifted, spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int npix, const int nbatch,
                          const int height, const int width,
                          const int nftrs, int max_nspix);


/************************************************************

        Split Marginal Likelihood + Hastings Functions

*************************************************************/

__global__
void calc_split_stats_step0_p(spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm_v2* sm_helper,
                              const int nspix_buffer,
                              float b_0, int max_nspix);

__global__
void calc_split_stats_step1_p(spix_params* sp_params,
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


                       Merge Proposal Functions


*************************************************************/


__global__
void calc_merge_candidate_p(int* seg, bool* border,
                            int* sm_pairs, spix_params* sp_params,
                            const int npix, const int nbatch,
                            const int width, const int height,
                            const int change);
__global__
void sum_by_label_merge_p(const float* img, const int* seg_gpu,
                          spix_params* sp_params,
                          spix_helper_sm_v2* sm_helper,
                          const int npix, const int nbatch,
                          const int width, const int nftrs);


__global__
void remove_sp_p(int* sm_pairs, spix_params* sp_params,
                 spix_helper_sm_v2* sm_helper,
                 const int nspix_buffer, int* nmerges);

__global__
void merge_sp_p(int* seg, bool* border, int* sm_pairs,
                spix_params* sp_params,
                spix_helper_sm_v2* sm_helper,
                const int npix, const int nbatch,
                const int width, const int height);


/************************************************************

                       Merge Functions

*************************************************************/

__global__
void calc_merge_stats_step0_p(int* sm_pairs, spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm_v2* sm_helper,
                              const int nspix_buffer, float b_0);

__global__
void calc_merge_stats_step1_p(spix_params* sp_params,
                              spix_helper_sm_v2* sm_helper,
                              const int nspix_buffer,
                              float a_0, float b_0);

__global__
void calc_merge_stats_step2_p(int* sm_pairs, spix_params* sp_params,
                              spix_helper_sm_v2* sm_helper,
                              const int nspix_buffer,
                              float alpha_hasting_ratio,
                              float merge_alpha);

__global__
void update_merge_flag_p(int* sm_pairs, spix_params* sp_params,
                         spix_helper_sm_v2* sm_helper,
                         const int nspix_buffer, int* nmerges);

