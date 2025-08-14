#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <cub/cub.cuh>

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> 
extract_unique_ids_batch_cub(
    int* spix_batch,            // Contiguous GPU memory: B×H×W
    thrust::device_vector<int>& prev_nspix,      // Array of previous superpixel counts
    int nbatch,             // Number of superpixel maps (e.g., 100)
    int npix                   // H × W (same for all batches)
);
                                                   
void compactify_new_superpixels_b(int* spix, spix_params* sp_params,
                            thrust::device_vector<int>& prop_ids,
			                //thrust::device_vector<int>& nspix,
			                thrust::device_vector<int>& new_nspix,
			                thrust::device_vector<int>& prev_nspix,
                            int nbatch, int nspix_buffer, int npix);


void launch_segmented_unique_2d(
    const thrust::device_vector<int>& sorted_spix_data,
    thrust::device_vector<int>& unique_spix_data,
    thrust::device_vector<int>& nspix,
    const thrust::device_vector<int>& segment_offsets,
    int nbatch, int npix);
    
__global__ void segmented_unique_kernel(
    const int* sorted_data,     // Input: sorted data [B*P elements]
    int* unique_data,           // Output: unique data [B*P elements] 
    int* counts,                // Output: count per batch [B elements]
    const int* segment_offsets, // Input: segment boundaries [B+1 elements]
    int nbatch,                 // Number of batches
    int npix                    // Elements per batch
);

 __global__ 
void compact_new_spix_b(int* spix, int* compression_map, int* prop_ids,
                        int* new_nspix, int* csum_new_nspix, int* prev_nspix_batch, int npix);

__global__ 
void fill_new_params_from_old_b(spix_params* params, spix_params*  new_params,
                              int* compression_map, int* csum_new_nspix, 
                              int* num_new_nspix, int nspix_buffer);

__global__ 
void fill_old_params_from_new_b(spix_params* params, spix_params*  new_params,
                                int* prev_max_batch, int* num_new_nspix, 
                                int* csum_new_nspix, int nspix_buffer);


