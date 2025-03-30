



#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"



__host__ int run_split_prop_v2(const float* img, int* seg, bool* border,
                       spix_params* sp_params, spix_helper* sp_helper,
                       spix_helper_sm* sm_helper,
                       int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                       float alpha_hastings,
                       float sigma2_app, float sigma2_size,
                       int& count, int idx, int max_nspix,
                       const int sp_size, const int npix,
                       const int nbatch, const int width,
                       const int height, const int nftrs,
                       const int nspix_buffer);

__host__ void run_merge_prop_v2(const float* img, int* seg, bool* border,
                        spix_params* sp_params, spix_helper* sp_helper,
                        spix_helper_sm* sm_helper,
                        int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                        float alpha_hastings,
                        float sigma2_app, float sigma2_size,
                        int& count, int idx, int max_nspix,
                        const int sp_size, const int npix,
                        const int nbatch, const int width,
                        const int height, const int nftrs,
                        const int nspix_buffer);

__host__ void CudaCalcMergeCandidate_prop_v2(const float* img, int* seg,
                                     bool* border, spix_params* sp_params,
                                     spix_helper* sp_helper,
                                     spix_helper_sm* sm_helper,
                                     int* sm_pairs, const int npix, const int nbatch,
                                     const int sp_size,
                                     const int width, const int height,
                                     const int nftrs, const int nspix_buffer,
                                     const int direction, float alpha,
                                     float sigma2_app, float sigma2_size);

__host__ int CudaCalcSplitCandidate_prop_v2(const float* img, int* seg, bool* border,
                                    spix_params* sp_params,
                                    spix_helper* sp_helper,
                                    spix_helper_sm* sm_helper,
                                    int* sm_seg1, int* sm_seg2, int* sm_pairs,
                                    const int sp_size,
                                    const int npix, const int nbatch, const int width,
                                    const int height, const int nftrs,
                                    const int nspix_buffer, int max_nspix,
                                    int direction,float alpha,
                                    float sigma2_app, float sigma2_size);

/************************************************************


                       Split Functions


*************************************************************/

__global__ void calc_split_stats_step0_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float b_0, int max_nspix);

__global__ void calc_split_stats_step1_p(spix_params* sp_params,
                                       spix_helper* sp_helper,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer,
                                       float a_0, float b_0, int max_nspix);

__global__
void update_split_flag_p(int* sm_pairs,
                       spix_params* sp_params,
                       spix_helper_sm* sm_helper,
                       const int nspix_buffer,
                       float alpha_hasting_ratio,
                       int max_nspix, int* max_sp );


/************************************************************


                       Merge Functions


*************************************************************/

__global__ void calc_merge_stats_step0_p(int* sm_pairs, spix_params* sp_params,
                                       spix_helper* sp_helper, spix_helper_sm* sm_helper,
                                       const int nspix_buffer, float b_0);
__global__ void calc_merge_stats_step1_p(spix_params* sp_params,spix_helper_sm* sm_helper,
                                       const int nspix_buffer,float a_0, float b_0);
__global__ void calc_merge_stats_step2_p(int* sm_pairs, spix_params* sp_params,
                                       spix_helper_sm* sm_helper,
                                       const int nspix_buffer, float alpha_hasting_ratio);
__global__ void update_merge_flag_p(int* sm_pairs, spix_params* sp_params,
                                  spix_helper_sm* sm_helper, const int nspix_buffer,
                                  int* nmerges);
