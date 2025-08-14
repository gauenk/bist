#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

// -- local --
#include "structs.h"
#include "init_utils.h"
#include "compact_spix_cub.h"
#define THREADS_PER_BLOCK 512

// // Add this right before your function call to test:
// // Replace the pragma message with this:
// #ifdef CUB_VERSION
// #error "CUB_VERSION is defined - this will show up as an error"
// #else  
// #error "CUB_VERSION is NOT defined - this will show up as an error"
// #endif



// Simple CUB-based batch processing for fixed-size H×W superpixel maps
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> 
extract_unique_ids_batch_cub(int* spix, thrust::device_vector<int>& prev_nspix, int nbatch, int npix) {
    
    // Step 1: Data is already contiguous! Just wrap it
    int total_pixels = nbatch * npix;
    thrust::device_ptr<int> spix_ptr(spix);
    thrust::device_vector<int> all_spix_data(spix_ptr, spix_ptr + total_pixels);
    
    // Step 2: Create segment offsets (all segments are size npix)
    thrust::device_vector<int> segment_offsets(nbatch + 1);
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nbatch + 1),
        segment_offsets.begin(),
        [npix] __device__(int i) { return i * npix; });
    
    // Step 3: CUB Segmented Sort - sort each H×W batch in parallel
    thrust::device_vector<int> sorted_spix_data(total_pixels);
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(
        nullptr, temp_storage_bytes,
        all_spix_data.data().get(), sorted_spix_data.data().get(), 
        total_pixels, nbatch,
        segment_offsets.data().get(), segment_offsets.data().get() + 1);
    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    cub::DeviceSegmentedSort::SortKeys(
        temp_storage.data().get(), temp_storage_bytes,
        all_spix_data.data().get(), sorted_spix_data.data().get(),
        total_pixels, nbatch,
        segment_offsets.data().get(), segment_offsets.data().get() + 1);
    
    // Step 4: CUB Segmented Unique - remove duplicates within each batch
    // Launch the kernel
    thrust::device_vector<int> unique_spix_data(total_pixels,-1);
    thrust::device_vector<int> nspix(nbatch,0);
    // dim3 grid(nbatch);
    // dim3 block(256);  // Adjust based on your npix
    // int num_blocks = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    // dim3 nblocks(num_blocks,nbatch);
    // dim3 nthreads(THREADS_PER_BLOCK);
    // int blocks_per_batch = (npix - 1) / threads_per_block + 1;  // Ceiling division
    // int elements_per_block = (npix - 1) / blocks_per_batch + 1;
    // size_t shared_mem_size = elements_per_block * sizeof(int);
    launch_segmented_unique_2d(
        sorted_spix_data,
        unique_spix_data,
        nspix,
        segment_offsets,
        nbatch, npix
    );
    
    // Step 5: Create B × S_max mapping matrix and track new superpixel counts
    int nspix_max = *thrust::max_element(nspix.begin(), nspix.end());
    thrust::device_vector<int> filtered_ids(nbatch * nspix_max, -1);
    thrust::device_vector<int> new_nspix(nbatch, 0);  // Track new superpixels per batch
    thrust::for_each(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nspix_max*nbatch),
        [unique_spix_ptr = thrust::raw_pointer_cast(unique_spix_data.data()),
         //unique_nspix_ptr = thrust::raw_pointer_cast(nspix.data()),
         prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data()),
         filtered_ids_ptr = thrust::raw_pointer_cast(filtered_ids.data()),
         nspix_ptr = thrust::raw_pointer_cast(nspix.data()),
         new_nspix_ptr = thrust::raw_pointer_cast(new_nspix.data()),
         nspix_max] __device__(int global_idx) {
            
            // global_index access each pixel across the BxHxW superpixel maps
            int batch_id = global_idx / nspix_max;
            int uniq_index = global_idx % nspix_max; 
            int _nspix = nspix_ptr[batch_id];

            if (uniq_index >= _nspix ) return; // Skip if index out of bounds

            // read unique spix id and previous nspix for this batch
            int spix_id = unique_spix_ptr[uniq_index];
            int _prev_nspix = prev_nspix_ptr[batch_id];
            if (spix_id < _prev_nspix) return; // Skip old/invalid IDs; already compacted

            // Calculate the new local index for this superpixel
            int store_index = atomicAdd(&new_nspix_ptr[batch_id], 1);
            int matrix_idx = store_index + nspix_max * batch_id;
            filtered_ids_ptr[matrix_idx] = spix_id;   
        });
    

    // Step 6: Go from B x NumUniq_max -> B x NumNew_max
    int num_new_max = *thrust::max_element(new_nspix.begin(), new_nspix.end());
    thrust::device_vector<int> filtered_ids_new(nbatch * num_new_max, -1);
    if (num_new_max == 0) {
        return std::make_pair(filtered_ids_new, new_nspix);
    }
    thrust::for_each(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nbatch * num_new_max),
        [filtered_ids_new_ptr = thrust::raw_pointer_cast(filtered_ids_new.data()),
         filtered_ids_ptr = thrust::raw_pointer_cast(filtered_ids.data()),
         new_nspix_ptr = thrust::raw_pointer_cast(new_nspix.data()),
         num_new_max, nspix_max] __device__(int global_idx) {
            
            // global_index access each pixel across the BxHxW superpixel maps
            int batch_id = global_idx / num_new_max;
            int new_index = global_idx % num_new_max;
            int num_new = new_nspix_ptr[batch_id];
            if (new_index >= num_new) return; // Skip if index out of bounds
            
            int pruned_index = new_index + num_new_max * batch_id;
            int full_index = new_index + nspix_max * batch_id;
            filtered_ids_new_ptr[pruned_index] = filtered_ids_ptr[full_index]; 
         });

    return std::make_tuple(filtered_ids_new, new_nspix);
}



// _spix,sp_params,prop_ids0,new_nspix_counts,nspix,nspix_prev,npix
void compactify_new_superpixels_b(int* spix, spix_params* sp_params,
                            thrust::device_vector<int>& prop_ids,
			                thrust::device_vector<int>& new_nspix,
			                thrust::device_vector<int>& prev_nspix, 
                            int nbatch, int nspix_buffer, int npix){

  // -- if no new spix, then skip compacting --
  int num_new_max = *thrust::max_element(new_nspix.begin(), new_nspix.end());
  if (num_new_max == 0) {return;}

  // -- easy indexing --
  thrust::device_vector<int> csum_new_nspix(new_nspix.size() + 1, 0);
  thrust::inclusive_scan(new_nspix.begin(), new_nspix.end(), csum_new_nspix.begin() + 1);
  int num_new_total = csum_new_nspix[nbatch];


  // -- update superpixel map and get simpler indexing tensor ['compression_map'] --
  int* compression_map=(int*)easy_allocate(num_new_total,sizeof(int));  
  thrust::for_each(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(nbatch * npix),
        [spix_ptr = spix, 
        prop_ids_ptr = thrust::raw_pointer_cast(prop_ids.data()),
        new_nspix_ptr = thrust::raw_pointer_cast(new_nspix.data()),        
        prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data()),
        csum_new_nspix_ptr = thrust::raw_pointer_cast(csum_new_nspix.data()),
        compression_map, npix, num_new_max] __device__(int global_idx) {
            int batch_id = global_idx / npix;
            int pix_index = global_idx % npix;
            int spix_id = spix_ptr[global_idx];
            int _prev_nspix = prev_nspix_ptr[batch_id];
            int _new_nspix = new_nspix_ptr[batch_id];
            int csum_offset = csum_new_nspix_ptr[batch_id];
            if (spix_id < _prev_nspix) return; // Skip old/invalid IDs; already compacted

            // Calculate the index for this superpixel
            for (int jx=0; jx < _new_nspix; jx++){
                int old_id = prop_ids_ptr[batch_id * num_new_max + jx];
                if (spix_id == old_id){
                    int new_spix_id = jx+_prev_nspix;
                    spix_ptr[global_idx] = jx+_prev_nspix; // update to "index" within "prop_ids" offset by prev max
                    compression_map[csum_offset+jx] = spix_id; // for updating spix_params
                    break;
                }
            }

        });
    
  // -- shift params into correct location --
  spix_params* new_params=(spix_params*)easy_allocate(num_new_total,sizeof(spix_params));
  int* prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data());
  int* num_new_ptr = thrust::raw_pointer_cast(new_nspix.data());
  int* csum_new_nspix_ptr = thrust::raw_pointer_cast(csum_new_nspix.data());
  int num_blocks1 = ceil( double(num_new_max) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks1(num_blocks1,nbatch);
  dim3 nthreads1(THREADS_PER_BLOCK);
  fill_new_params_from_old_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                     compression_map,num_new_ptr,
                                                     csum_new_nspix_ptr,nspix_buffer);
  fill_old_params_from_new_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                     prev_nspix_ptr,num_new_ptr,
                                                     csum_new_nspix_ptr,nspix_buffer);

  // -- free parameters --
  cudaFree(new_params);
  cudaFree(compression_map);

// (spix_params* params, spix_params*  new_params,
//                                 int* compression_map, int* num_new_nspix, int* csum_new_nspix)
  return;
}


// Host function to launch the kernel with your preferred launch parameters
void launch_segmented_unique_2d(
    const thrust::device_vector<int>& sorted_spix_data,
    thrust::device_vector<int>& unique_spix_data,
    thrust::device_vector<int>& nspix,
    const thrust::device_vector<int>& segment_offsets,
    int nbatch, int npix) {
    
    // Your launch parameters
    int num_blocks = ceil(double(npix) / double(THREADS_PER_BLOCK));
    dim3 nblocks(num_blocks, nbatch);
    dim3 nthreads(THREADS_PER_BLOCK);
    
    int blocks_per_batch = (npix - 1) / THREADS_PER_BLOCK + 1;  // Your ceiling division
    int elements_per_block = (npix - 1) / blocks_per_batch + 1;
    size_t shared_mem_size = elements_per_block * sizeof(int);
    
    // Launch kernel with your parameters
    segmented_unique_kernel<<<nblocks, nthreads, shared_mem_size>>>(
        thrust::raw_pointer_cast(sorted_spix_data.data()),
        thrust::raw_pointer_cast(unique_spix_data.data()),
        thrust::raw_pointer_cast(nspix.data()),
        thrust::raw_pointer_cast(segment_offsets.data()),
        nbatch, npix
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}


__global__ void segmented_unique_kernel(
    const int* sorted_data,     // Input: sorted data [B*P elements]
    int* unique_data,           // Output: unique data [B*P elements] 
    int* counts,                // Output: count per batch [B elements]
    const int* segment_offsets, // Input: segment boundaries [B+1 elements]
    int nbatch,                 // Number of batches
    int npix                    // Elements per batch
) {
    // With 2D grid: blockIdx.x = block within batch, blockIdx.y = batch index
    int batch_idx = blockIdx.y;
    int block_in_batch = blockIdx.x;
    
    // Get start and end indices for this batch
    int batch_start = segment_offsets[batch_idx];
    int batch_end = segment_offsets[batch_idx + 1];
    int segment_size = batch_end - batch_start;
    assert(segment_size == npix); // Ensure each batch has npix elements
    
    // Calculate how many blocks are processing this batch
    int blocks_per_batch = gridDim.x;
    
    // Shared memory counter for unique elements found by THIS block
    __shared__ int block_unique_count;
    __shared__ int block_write_offset;  // Where this block should write its results
    
    if (threadIdx.x == 0) {
        block_unique_count = 0;
    }
    __syncthreads();
    
    // Phase 1: Each block processes its portion of the batch
    int elements_per_block = (segment_size + blocks_per_batch - 1) / blocks_per_batch;
    int block_start = batch_start + block_in_batch * elements_per_block;
    int block_end = min(block_start + elements_per_block, batch_end);
    
    // Temporary storage for this block's unique elements
    extern __shared__ int temp_unique[];
    
    // Find unique elements in this block's portion
    for (int local_idx = threadIdx.x; local_idx < (block_end - block_start); local_idx += blockDim.x) {
        int global_idx = block_start + local_idx;
        
        bool is_unique = false;
        
        if (global_idx == batch_start) {
            // First element of entire batch is always unique
            is_unique = true;
        } else {
            // Check if different from previous element
            if (sorted_data[global_idx] != sorted_data[global_idx - 1]) {
                is_unique = true;
            }
        }
        
        if (is_unique) {
            // Store in block's temporary array
            int temp_pos = atomicAdd(&block_unique_count, 1);
            temp_unique[temp_pos] = sorted_data[global_idx];
        }
    }
    
    __syncthreads();
    
    // Phase 2: Coordinate between blocks to get write positions
    // Use global atomic to reserve space for this block's unique elements
    if (threadIdx.x == 0) {
        block_write_offset = atomicAdd(&counts[batch_idx], block_unique_count);
    }
    __syncthreads();
    
    // Phase 3: Write this block's unique elements to final output
    for (int i = threadIdx.x; i < block_unique_count; i += blockDim.x) {
        unique_data[batch_start + block_write_offset + i] = temp_unique[i];
    }
}



__global__ 
void compact_new_spix_b(int* spix, int* compression_map, int* prop_ids,
                        int* new_nspix, int* csum_new_nspix, int* prev_nspix, int npix){

  // -- indexing pixel indices --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= npix) return;
  int bi = blockIdx.y;
  int _prev_nspix = prev_nspix[bi];
  int num_new = new_nspix[bi];
  int spix_offset = csum_new_nspix[bi];
  ix = bi * npix + ix; // global index for this batch

  // -- read current id --
  int spix_id = spix[ix];
  if (spix_id < _prev_nspix){ return; } // numbering starts @ 0; so "=prev_nspix" is new

  // -- update to compact index if "new" --
  // int shift_ix = spix_id - prev_nspix;
  for (int jx=0; jx < num_new; jx++){
    if (spix_id == prop_ids[jx]){
      int new_spix_id = jx+_prev_nspix;
      spix[ix] = jx+_prev_nspix; // update to "index" within "prop_ids" offset by prev max
      compression_map[jx] = spix_id; // for updating spix_params
      break;
    }
  }
}

__global__ 
void fill_new_params_from_old_b(spix_params* params, spix_params*  new_params,
                                int* compression_map, int* num_new_nspix, int* csum_new_nspix,
                                int nspix_buffer){

  // -- indexing new superpixel labels --
  int dest_ix = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_ix = blockIdx.y;
  int num_new = num_new_nspix[batch_ix];
  int csum_offset = csum_new_nspix[batch_ix];
  if (dest_ix >= num_new) return;
  dest_ix = dest_ix + csum_offset;

  int label_offset = batch_ix*nspix_buffer;
  int src_ix = compression_map[dest_ix]+label_offset;

  int valid_src = params[src_ix].valid ?  1 : 0;
  int valid_dest = params[dest_ix].valid ?  1 : 0;
  // if (src_ix==30){
  //   printf("[fill] src_ix,dest_ix: %d,%d [%d,%d]\n",src_ix,dest_ix,valid_src,valid_dest);
  // }
  // if (dest_ix==30){
  //   printf("[fill] src_ix,dest_ix: %d,%d [%d,%d]\n",src_ix,dest_ix,valid_src,valid_dest);
  // }
  // if (dest_ix==42){
  //   printf("[fill] src_ix,dest_ix: %d,%d [%d,%d]\n",src_ix,dest_ix,valid_src,valid_dest);
  // }

  // new_params[dest_ix] = params[src_ix];

  new_params[dest_ix].mu_app = params[src_ix].mu_app;
  // new_params[dest_ix].sigma_app = params[src_ix].sigma_app;
  new_params[dest_ix].prior_mu_app = params[src_ix].prior_mu_app;
  // new_params[dest_ix].prior_sigma_app = params[src_ix].prior_sigma_app;
  new_params[dest_ix].prior_mu_app_count = params[src_ix].prior_mu_app_count;
  // new_params[dest_ix].prior_sigma_app_count = params[src_ix].prior_sigma_app_count;
  new_params[dest_ix].mu_shape = params[src_ix].mu_shape;
  new_params[dest_ix].sigma_shape = params[src_ix].sigma_shape;
  new_params[dest_ix].prior_mu_shape = params[src_ix].prior_mu_shape;
  new_params[dest_ix].prior_sigma_shape = params[src_ix].prior_sigma_shape;
  new_params[dest_ix].sample_sigma_shape = params[src_ix].sample_sigma_shape;
  new_params[dest_ix].prior_mu_shape_count = params[src_ix].prior_mu_shape_count;
  new_params[dest_ix].prior_sigma_shape_count = params[src_ix].prior_sigma_shape_count;
  // new_params[dest_ix].logdet_sigma_app = params[src_ix].logdet_sigma_app;
  new_params[dest_ix].logdet_sigma_shape = params[src_ix].logdet_sigma_shape;
  // new_params[dest_ix].logdet_prior_sigma_shape = params[src_ix].logdet_prior_sigma_shape;
  // new_params[dest_ix].prior_lprob = params[src_ix].prior_lprob;
  new_params[dest_ix].count = params[src_ix].count;
  new_params[dest_ix].prior_count = params[src_ix].prior_count;
  new_params[dest_ix].valid = params[src_ix].valid;

}

__global__ 
void fill_old_params_from_new_b(spix_params* params, spix_params*  new_params,
                                int* prev_nspix, int* new_nspix, int* csum_new_nspix,
                                int nspix_buffer){

  // -- indexing new superpixel labels --
  int _ix = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = blockIdx.y;
  int _prev_nspix = prev_nspix[bi];
  int _new_nspix = new_nspix[bi];
  int csum_offset = csum_new_nspix[bi];
  if (_ix >= _new_nspix) return; // num_new
  int jx = _ix + csum_offset;
  int ix = _ix + _prev_nspix + nspix_buffer * bi; // global index for this batch
  params[ix].mu_app = new_params[jx].mu_app;
  // params[ix].sigma_app = new_params[jx].sigma_app;
  params[ix].prior_mu_app = new_params[jx].prior_mu_app;
  // params[ix].prior_sigma_app = new_params[jx].prior_sigma_app;
  params[ix].prior_mu_app_count = new_params[jx].prior_mu_app_count;
  // params[ix].prior_sigma_app_count = new_params[jx].prior_sigma_app_count;
  params[ix].mu_shape = new_params[jx].mu_shape;
  params[ix].sigma_shape = new_params[jx].sigma_shape;
  params[ix].prior_mu_shape = new_params[jx].prior_mu_shape;
  params[ix].prior_sigma_shape = new_params[jx].prior_sigma_shape;
  params[ix].sample_sigma_shape = new_params[jx].sample_sigma_shape;
  params[ix].prior_mu_shape_count = new_params[jx].prior_mu_shape_count;
  params[ix].prior_sigma_shape_count = new_params[jx].prior_sigma_shape_count;
  // params[ix].logdet_sigma_app = new_params[jx].logdet_sigma_app;
  params[ix].logdet_sigma_shape = new_params[jx].logdet_sigma_shape;
  params[ix].logdet_prior_sigma_shape = new_params[jx].logdet_prior_sigma_shape;
  // params[ix].prior_lprob = new_params[jx].prior_lprob;
  params[ix].count = new_params[jx].count;
  params[ix].prior_count = new_params[jx].prior_count;
  params[ix].valid = new_params[jx].valid;

}

