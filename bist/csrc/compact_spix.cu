

/********************************************************************

              Ensure the New Superpixels "LABEL" Are Compact

   For example: [0,1,2] + [3,5,6] -> [0,1,2] + [3,4,5]


   -- ensure the new spix are added without including empty spix --
   i don't think its serious to keep these,
   but conceptually it may be confusing when analyzing the output

   the only danger would be not-initialized PySuperpixelParams...
    so maybe this danger is real...



********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <assert.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- local --
#include "structs.h"
#include "init_utils.h"
#include "compact_spix.h"
#define THREADS_PER_BLOCK 512
// #include "sparams_io.h"
// #include "init_sparams.h"



int compactify_new_superpixels(int* spix, spix_params* sp_params,
                               thrust::device_vector<int>& prop_ids,
                               int prev_nspix,int max_spix,int npix){

  // -- get new ids --
  // auto unique_ids = std::get<0>(at::_unique(spix));
  // auto ids = unique_ids.data<int>();
  // int num_ids = unique_ids.sizes()[0];
  // auto mask = unique_ids >= prev_nspix;
  // auto prop_ids = unique_ids.masked_select(mask); // uncompressed and alive
  // thrust::device_vector<int> prop_ids = extract_unique_ids(spix, npix, prev_nspix);

  // // -- if no new spix, then skip compacting --
  // if ((max_spix+1) == prev_nspix)

  // -- update maximum number of superpixels --
  int num_new = prop_ids.size();
  int compact_nspix = prev_nspix + num_new;
  // fprintf(stdout,"prev_nspix: %d\n",prev_nspix);
  // fprintf(stdout,"num_new: %d\n",num_new);
  if (num_new == 0) {return compact_nspix;}

  // -- allocate spix for storing --
  spix_params* new_params=(spix_params*)easy_allocate(num_new,sizeof(spix_params));
  int* compression_map=(int*)easy_allocate(num_new,sizeof(int));

  // -- update spix labels and params to reflect compacted labeling --
  int* prop_ids_ptr = thrust::raw_pointer_cast(prop_ids.data());
  int num_blocks = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks(num_blocks);
  dim3 nthreads(THREADS_PER_BLOCK);
  compact_new_spix<<<nblocks,nthreads>>>(spix,compression_map,
                                         prop_ids_ptr,num_new,prev_nspix,npix);

  // -- shift params into correct location --
  int num_blocks1 = ceil( double(num_new) / double(THREADS_PER_BLOCK) ); 
  dim3 nblocks1(num_blocks1);
  dim3 nthreads1(THREADS_PER_BLOCK);
  fill_new_params_from_old<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                  compression_map,num_new);
  fill_old_params_from_new<<<nblocks1,nthreads1>>>(sp_params,new_params,
                                                  prev_nspix,num_new);

  // -- free parameters --
  cudaFree(compression_map);
  cudaFree(new_params);

  return compact_nspix;
}


// Subroutine to extract unique IDs
thrust::device_vector<int> extract_unique_ids(int* spix, int npix, int prev_nspix) {

    // Wrap the raw pointer with a thrust::device_ptr
    thrust::device_ptr<int> spix_ptr(spix);

    // Step 1: Copy the raw pointer into a thrust::device_vector
    thrust::device_vector<int> spix_vector(spix_ptr, spix_ptr + npix);

    // Step 2: Sort the vector
    thrust::sort(spix_vector.begin(), spix_vector.end());

    // Step 3: Remove duplicates
    auto end_unique = thrust::unique(spix_vector.begin(), spix_vector.end());

    // Step 4: Resize to keep only unique elements
    spix_vector.erase(end_unique, spix_vector.end());
    // printf("num unique: %d\n",spix_vector.size());

    // Step 5: TODO ensure only 
    thrust::device_vector<int> filtered_ids(spix_vector.size());
    auto end_filtered = thrust::copy_if(
        spix_vector.begin(), spix_vector.end(), filtered_ids.begin(),
        [=] __device__(int x) { return x >= prev_nspix; });

    // Resize filtered_ids to the number of retained elements
    filtered_ids.erase(end_filtered, filtered_ids.end());

    // The vector now contains only unique IDs
    return filtered_ids;
}

thrust::device_vector<int> remove_old_ids(thrust::device_vector<int>& spix_vector,
                                          int prev_nspix) {
    // // Sort
    // thrust::sort(spix_vector.begin(), spix_vector.end());

    // Step 5: TODO ensure only 
    thrust::device_vector<int> filtered_ids(spix_vector.size());
    auto end_filtered = thrust::copy_if(
        spix_vector.begin(), spix_vector.end(), filtered_ids.begin(),
        [=] __device__(int x) { return x >= prev_nspix; });

    // Resize filtered_ids to the number of retained elements
    filtered_ids.erase(end_filtered, filtered_ids.end());
    return filtered_ids;
}



// thrust::device_vector<int>
// get_filtered_ids(thrust::device_vector<int> ids, int prev_spix){

//     // Step 5: TODO ensure only 
//     auto end_filtered = thrust::copy_if(
//         ids.begin(), ids.end(), ids.begin(),
//         [=] __device__(int x) { return x >= prev_nspix; });

//     // Resize filtered_ids to the number of retained elements
//     ids.erase(end_filtered, filtered_ids.end());

// //     // The vector now contains only unique IDs
// //     return ids;
// // }








































__global__ 
void compact_new_spix(int* spix, int* compression_map, int* prop_ids,
                      int num_new, int prev_nspix, int npix){

  // -- indexing pixel indices --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= npix) return;

  // -- read current id --
  int spix_id = spix[ix];
  if (spix_id < prev_nspix){ return; } // numbering starts @ 0; so "=prev_nspix" is new

  // -- update to compact index if "new" --
  // int shift_ix = spix_id - prev_nspix;
  for (int jx=0; jx < num_new; jx++){
    if (spix_id == prop_ids[jx]){
      int new_spix_id = jx+prev_nspix;
      spix[ix] = jx+prev_nspix; // update to "index" within "prop_ids" offset by prev max
      compression_map[jx] = spix_id; // for updating spix_params
      break;
    }
  }
}

__global__ 
void fill_new_params_from_old(spix_params* params, spix_params*  new_params,
                              int* compression_map, int num_new){

  // -- indexing new superpixel labels --
  int dest_ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (dest_ix >= num_new) return;
  int src_ix = compression_map[dest_ix];

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
void fill_old_params_from_new(spix_params* params, spix_params*  new_params,
                              int prev_max, int num_new){

  // -- indexing new superpixel labels --
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix >= num_new) return;
  params[prev_max+ix].mu_app = new_params[ix].mu_app;
  // params[prev_max+ix].sigma_app = new_params[ix].sigma_app;
  params[prev_max+ix].prior_mu_app = new_params[ix].prior_mu_app;
  // params[prev_max+ix].prior_sigma_app = new_params[ix].prior_sigma_app;
  params[prev_max+ix].prior_mu_app_count = new_params[ix].prior_mu_app_count;
  // params[prev_max+ix].prior_sigma_app_count = new_params[ix].prior_sigma_app_count;
  params[prev_max+ix].mu_shape = new_params[ix].mu_shape;
  params[prev_max+ix].sigma_shape = new_params[ix].sigma_shape;
  params[prev_max+ix].prior_mu_shape = new_params[ix].prior_mu_shape;
  params[prev_max+ix].prior_sigma_shape = new_params[ix].prior_sigma_shape;
  params[prev_max+ix].sample_sigma_shape = new_params[ix].sample_sigma_shape;
  params[prev_max+ix].prior_mu_shape_count = new_params[ix].prior_mu_shape_count;
  params[prev_max+ix].prior_sigma_shape_count = new_params[ix].prior_sigma_shape_count;
  // params[prev_max+ix].logdet_sigma_app = new_params[ix].logdet_sigma_app;
  params[prev_max+ix].logdet_sigma_shape = new_params[ix].logdet_sigma_shape;
  params[prev_max+ix].logdet_prior_sigma_shape = new_params[ix].logdet_prior_sigma_shape;
  // params[prev_max+ix].prior_lprob = new_params[ix].prior_lprob;
  params[prev_max+ix].count = new_params[ix].count;
  params[prev_max+ix].prior_count = new_params[ix].prior_count;
  params[prev_max+ix].valid = new_params[ix].valid;

}

