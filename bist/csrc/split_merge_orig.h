


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"



__host__ int run_split_orig(const float* img, int* seg, bool* border,
                            spix_params* sp_params, spix_helper* sp_helper,
                            spix_helper_sm* sm_helper,
                            int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                            float alpha_hastings, float split_alpha,
                            float sigma2_app, float sigma2_size,
                            int& count, int idx, int max_nspix,
                            const int sp_size, const int npix,
                            const int nbatch, const int width,
                            const int height, const int nftrs,
                            const int nspix_buffer, Logger* logger=nullptr);

__host__ void run_merge_orig(const float* img, int* seg, bool* border,
                             spix_params* sp_params, spix_helper* sp_helper,
                             spix_helper_sm* sm_helper,
                             int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                             float alpha_hastings, float alpha_merge,
                             float sigma2_app, float sigma2_size,
                             int& count, int idx, int max_nspix,
                             const int sp_size, const int npix,
                             const int nbatch, const int width,
                             const int height, const int nftrs,
                             const int nspix_buffer, Logger* logger=nullptr);

__host__ void CudaCalcMergeCandidate_orig(const float* img, int* seg,
                                     bool* border, spix_params* sp_params,
                                     spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,
                                     int* sm_pairs, const int npix, const int nbatch,
                                     const int sp_size,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha, float alpha_merge,
                                     float sigma2_app, float sigma2_size, Logger*  logger=nullptr);

__host__ int CudaCalcSplitCandidate_orig(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int sp_size,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_nspix,
                                    int direction,float alpha,float split_alpha,
                                    float sigma2_app, float sigma2_size, Logger* logger=nullptr);

/************************************************************


                       Split Functions


*************************************************************/

__global__ void calc_split_stats_step0(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float b_0, int max_nspix);

__global__ void calc_split_stats_step1(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float a_0, float b_0, int max_nspix);

__global__
void update_split_flag(int* sm_pairs,
                       spix_params* sp_params,
                       spix_helper_sm* sm_helper,
                       const int nspix_buffer,
                       float alpha_hasting_ratio,
                       float split_alpha,
                       int max_nspix, int* max_sp );


/************************************************************


                       Merge Functions


*************************************************************/

__global__ void calc_merge_stats_step0(int* sm_pairs, spix_params* sp_params,
                                       spix_helper* sp_helper, spix_helper_sm* sm_helper,
                                       const int nspix_buffer, float b_0);
__global__ void calc_merge_stats_step1(spix_params* sp_params,spix_helper_sm* sm_helper,
                                       const int nspix_buffer,float a_0, float b_0);
__global__ void calc_merge_stats_step2(int* sm_pairs, spix_params* sp_params,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float alpha_hasting_ratio, float alpha_merge);
__global__ void update_merge_flag(int* sm_pairs, spix_params* sp_params,
                                  spix_helper_sm* sm_helper, const int nspix_buffer,
                                  int* nmerges);
