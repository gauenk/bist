

// -- cuda --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>


int compactify_new_superpixels(int* spix, spix_params* sp_params,
                               thrust::device_vector<int>& prop_ids,
                               int prev_nspix,int max_spix,int npix);
void get_compact_spix(int* spix, int npix);

thrust::device_vector<int> extract_unique_ids(int* spix, int max_spix, int prev_nspix);
thrust::device_vector<int> remove_old_ids(thrust::device_vector<int>& spix_vector, int prev_nspix);

__global__ void compact_new_spix(int* spix, int* compression_map, int* prop_ids,
                                 int num_new, int prev_nspix, int npix);
__global__ void fill_new_params_from_old(spix_params* params, spix_params*  new_params,
                                         int* compression_map, int num_new);
__global__ void fill_old_params_from_new(spix_params* params, spix_params*  new_params,
                                         int prev_max, int num_new);
__global__ void compact_spix_kernel(int* spix, int* uniq, int npix, int nuniq);
