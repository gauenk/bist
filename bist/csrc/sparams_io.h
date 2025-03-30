
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"

SuperpixelParams* get_params_as_vectors(spix_params* sp_params, int* ids,
                                         int num, int max_num);

__host__
spix_params* get_vectors_as_params(SuperpixelParams* params, int sp_size,
                                   int npix, int nspix,int nspix_buffer);

__host__
SuperpixelParams* get_output_params(spix_params* sp_params,
                                     SuperpixelParams* prior_params,
                                     int* ids,int num, int max_num);

__host__
void fill_params_from_params(SuperpixelParams* dest_params,
                             SuperpixelParams* src_params);

__host__
void params_to_vectors(SuperpixelParams* sp_params_py,
                       spix_params* sp_params, int* ids, int num);

__host__
void vectors_to_params(SuperpixelParams* sp_params_py,
                       spix_params* sp_params);


__global__
void read_params(float* mu_app, float* prior_mu_app,
                 double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                 double* prior_mu_shape, double* prior_sigma_shape,
                 double* sample_sigma_shape,
                 int* counts, float* icounts, int* sm_counts, float* prior_counts,
                 spix_params* sp_params, int* ids, int spix);

__global__
void write_params(float* mu_app, float* prior_mu_app,
                  double* mu_shape, double* sigma_shape, float* logdet_sigma_shape,
                  double* prior_mu_shape, double* prior_sigma_shape,
                  double* sample_sigma_shape,
                  int* counts, float* icounts, int* sm_counts, float* prior_counts,
                  spix_params* sp_params, int nspix);
__host__
void write_prior_counts(SuperpixelParams* src_params,spix_params* dest_params,
                        int* ids, int nactive);

__device__
double3 invert_cov(double3 cov);

__device__
double3 get_cov_eigenvals(double3 cov);

__global__
void write_prior_counts_kernel(float* prior_counts, spix_params* sp_params,
                               int* ids, int nactive);

/* void run_update_prior(thrust::device_vector<int>& spix, */
/*                       SuperpixelParams* params, int max_spix, bool invert); */
void run_update_prior(SuperpixelParams* params, int* ids,
                      int npix, int nspix, int prev_nspix, bool invert);

__global__
void update_prior_kernel(float* mu_app, float* prior_mu_app,
                         double* mu_shape, double* prior_mu_shape,
                         double* sigma_shape, double* prior_sigma_shape,
                         double* sample_sigma_shape,
                         int* counts, int* ids, int nspix, int prev_max_spix, bool invert);


/* __host__ */
/* SuperpixelParams* get_params_as_tensors(spix_params* sp_params, */
/*                                          int* ids, int num); */

/* __host__ */
/* void write_prior_counts(SuperpixelParams* src_params,spix_params* dest_params); */


// -- compact --
/* __global__ */
/* void compact_new_spix(int* spix, int* compression_map, int* prop_ids, */
/*                       int num_new, int prev_max, int npix); */
/* __global__ */
/* void fill_new_params_from_old(spix_params* params, spix_params*  new_params, */
/*                               int* compression_map, int num_new); */
/* __global__ */
/* void fill_old_params_from_new(spix_params* params, spix_params*  new_params, */
/*                               int prev_max, int num_new); */
/* int compactify_new_superpixels(torch::Tensor spix,spix_params* sp_params, */
/*                                int prev_max_spix,int max_spix,int npix); */


/* __host__ SuperpixelParams* init_tensor_params(int size); */

