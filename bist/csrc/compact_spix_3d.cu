



// -- nice cumsum --
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>

// -- local imports --
#include "compact_spix_3d.h"
#include "init_utils.h"

#define THREADS_PER_BLOCK 512


// Alternative version with better initialization handling
__global__ void count_and_assign_labels(
    uint32_t* labels,
    const uint8_t* bids,
    uint32_t* nspix,
    uint32_t* nspix_old,
    uint32_t* max_new_csum,
    uint32_t* label2index_shell,
    uint32_t* relabel_map,
    int nbatch, int nnodes, int S) {

    // one index per node
    int node_ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_ix >= nnodes) return;
    int bx = bids[node_ix];
    uint32_t label = labels[node_ix];
    uint32_t _nspix_old = nspix_old[bx];
    assert(_nspix_old == 0); // dev only presently
    if (label < _nspix_old) { // skip me!
        return;
    }
   
    // Claim this slot
    int shell_index = (label - _nspix_old) + max_new_csum[bx];
    if (shell_index >= S){
        printf("S: %d, shell_index: %d\n",S,shell_index);
    }
    assert(shell_index < S);
    uint32_t old_label = atomicCAS(
        (unsigned int*)&label2index_shell[shell_index],
        UINT32_MAX, (unsigned int)label
    );
    unsigned int one = 1;
    unsigned int zero = 0;
    if (old_label == UINT32_MAX) {
        // Successfully claimed slot - assign new label
        unsigned int new_label = atomicAdd((unsigned int*)&nspix[bx],one);
        
        // Atomically set the label (using -1 as sentinel for "being written")
        atomicExch((unsigned int*)&relabel_map[shell_index], new_label);
        labels[node_ix] = new_label;
        return;
        
    } else if (old_label == label) {
        // Hash exists - spin until label is available
        unsigned int _label;
        int spin_count = 0;
        do {
            _label = atomicAdd((unsigned int*)&relabel_map[shell_index], zero); // Atomic read
            //if (_label >= 0) break;
            if( _label != UINT32_MAX) break;
            
            // Yield occasionally to prevent warp divergence issues
            if (++spin_count > 100) {
                __nanosleep(1); // Small delay if available
                spin_count = 0;
            }
        // } while (_label < 0);
        } while (_label == UINT32_MAX);

        labels[node_ix] = _label;
        return;
    }   

}


struct diff_op {
    __host__ __device__
    uint32_t operator()(const thrust::tuple<uint32_t,uint32_t>& t) const {
        return thrust::get<1>(t) - thrust::get<0>(t);
    }
};

void run_compactify(uint32_t* nspix, uint32_t* spix, uint8_t* bids, uint32_t* nspix_old, uint32_t* max_new_nspix, int nbatch, int nnodes) {

    // get the cummulative sum of the differences
    uint32_t* max_new_csum = (uint32_t*)easy_allocate((nbatch+1),sizeof(uint32_t));
    {
        cudaMemset(max_new_csum, 0, sizeof(uint32_t));
        thrust::device_ptr<uint32_t> in0_ptr(nspix_old);
        thrust::device_ptr<uint32_t> in1_ptr(max_new_nspix);
        thrust::device_ptr<uint32_t> out_ptr(max_new_csum);
        auto first = thrust::make_zip_iterator(thrust::make_tuple(in0_ptr, in1_ptr));
        auto last  = thrust::make_zip_iterator(thrust::make_tuple(in0_ptr + nbatch, in1_ptr + nbatch));
        // Inclusive scan over the difference, writing to out[1..N]
        thrust::inclusive_scan(
            thrust::device,
            thrust::make_transform_iterator(first, diff_op()),
            thrust::make_transform_iterator(last,  diff_op()),
            out_ptr + 1
        );
    }
    uint32_t max_num_new;
    cudaMemcpy(&max_num_new,&max_new_csum[nbatch],sizeof(uint32_t),cudaMemcpyDeviceToHost);
    printf("max_new_csum[nbatch]: %d\n",max_num_new);
    
    // Step 2: Count unique hashes and create label map
    //ntotal_nspix = cusmum(nspix)[-1];
    // since i already know the nspix going into this... maybe this can be an input and a derived quantity at the input so its more compact?
    // i think a reason why hashmap is so big is because we didn't know how many unique values there were... but now we know this at the init...
    // but actually we dont...
    // ...
    // after init i don't think there is any problem since "sum of max nspix per batch" would be pretty close to the actual number of superpixels...
    // yeah pretty minor after init...
    // but the init case is just sooo bad... well for now we spike it i guess.
    //uint32_t* label_map = easy_allocate(max_num_new,sizeof(uint32_t));
    // uint32_t* nspix = (uint32_t*)easy_allocate(nbatch,sizeof(uint32_t));
    cudaMemcpy(nspix,nspix_old,nbatch*sizeof(uint32_t),cudaMemcpyDeviceToDevice);
    uint32_t* label2index_shell = (uint32_t*)easy_allocate(max_num_new,sizeof(uint32_t));
    cudaMemset(label2index_shell, 0xFF, max_num_new * sizeof(uint32_t));
    //uint32_t* relabel_bids = easy_allocate(max_num_new,sizeof(uint32_t));
    uint32_t* relabel_map = (uint32_t*)easy_allocate(max_num_new,sizeof(uint32_t));
    cudaMemset(relabel_map, 0xFF, max_num_new * sizeof(uint32_t));
    // todo; relabel_map init to "-1"
   
    printf("max_num_new: %d\n",max_num_new);
    dim3 count_block(256);
    dim3 count_grid((nnodes + count_block.x - 1) / count_block.x);
    count_and_assign_labels<<<count_grid, count_block>>>(
        spix,bids,
        nspix,nspix_old,max_new_csum,
        label2index_shell,relabel_map,
        nbatch, nnodes, max_num_new);

    // -- shift superpixels --
    // // -- shift params into correct location --
    // spix_params* new_params=(spix_params*)easy_allocate(num_new_total,sizeof(spix_params));
    // int* prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data());
    // int* num_new_ptr = thrust::raw_pointer_cast(new_nspix.data());
    // int* csum_new_nspix_ptr = thrust::raw_pointer_cast(csum_new_nspix.data());
    // int num_blocks1 = ceil( double(num_new_max) / double(THREADS_PER_BLOCK) ); 
    // dim3 nblocks1(num_blocks1,nbatch);
    // dim3 nthreads1(THREADS_PER_BLOCK);
    // fill_new_params_from_old_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
    //                                                     compression_map,num_new_ptr,
    //                                                     csum_new_nspix_ptr,nspix_buffer);
    // fill_old_params_from_new_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
    //                                                     prev_nspix_ptr,num_new_ptr,
    //                                                     csum_new_nspix_ptr,nspix_buffer);

    // -- free parameters --
    cudaFree(relabel_map);
    cudaFree(label2index_shell);
    cudaFree(max_new_csum);

    return;
}




// // _spix,sp_params,prop_ids0,new_nspix_counts,nspix,nspix_prev,npix
// void compactify_new_superpixels_3d(int* spix, spix_params* sp_params,
//                                 thrust::device_vector<int>& prop_ids,
// 			                    thrust::device_vector<int>& new_nspix,
// 			                    thrust::device_vector<int>& prev_nspix, 
//                                 int nbatch, int nspix_buffer, int npix){

//   // -- if no new spix, then skip compacting --
//   int num_new_max = *thrust::max_element(new_nspix.begin(), new_nspix.end());
//   if (num_new_max == 0) {return;}

//   // -- easy indexing --
//   thrust::device_vector<int> csum_new_nspix(new_nspix.size() + 1, 0);
//   thrust::inclusive_scan(new_nspix.begin(), new_nspix.end(), csum_new_nspix.begin() + 1);
//   int num_new_total = csum_new_nspix[nbatch];


//   // -- update superpixel map and get simpler indexing tensor ['compression_map'] --
//   int* compression_map=(int*)easy_allocate(num_new_total,sizeof(int));  
//   thrust::for_each(
//     thrust::counting_iterator<int>(0),
//     thrust::counting_iterator<int>(nbatch * npix),
//         [spix_ptr = spix, 
//         prop_ids_ptr = thrust::raw_pointer_cast(prop_ids.data()),
//         new_nspix_ptr = thrust::raw_pointer_cast(new_nspix.data()),        
//         prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data()),
//         csum_new_nspix_ptr = thrust::raw_pointer_cast(csum_new_nspix.data()),
//         compression_map, npix, num_new_max] __device__(int global_idx) {
//             int batch_id = global_idx / npix;
//             int pix_index = global_idx % npix;
//             int spix_id = spix_ptr[global_idx];
//             int _prev_nspix = prev_nspix_ptr[batch_id];
//             int _new_nspix = new_nspix_ptr[batch_id];
//             int csum_offset = csum_new_nspix_ptr[batch_id];
//             if (spix_id < _prev_nspix) return; // Skip old/invalid IDs; already compacted

//             // Calculate the index for this superpixel
//             for (int jx=0; jx < _new_nspix; jx++){
//                 int old_id = prop_ids_ptr[batch_id * num_new_max + jx];
//                 if (spix_id == old_id){
//                     int new_spix_id = jx+_prev_nspix;
//                     spix_ptr[global_idx] = jx+_prev_nspix; // update to "index" within "prop_ids" offset by prev max
//                     compression_map[csum_offset+jx] = spix_id; // for updating spix_params
//                     break;
//                 }
//             }

//         });
    
//   // -- shift params into correct location --
//   spix_params* new_params=(spix_params*)easy_allocate(num_new_total,sizeof(spix_params));
//   int* prev_nspix_ptr = thrust::raw_pointer_cast(prev_nspix.data());
//   int* num_new_ptr = thrust::raw_pointer_cast(new_nspix.data());
//   int* csum_new_nspix_ptr = thrust::raw_pointer_cast(csum_new_nspix.data());
//   int num_blocks1 = ceil( double(num_new_max) / double(THREADS_PER_BLOCK) ); 
//   dim3 nblocks1(num_blocks1,nbatch);
//   dim3 nthreads1(THREADS_PER_BLOCK);
//   fill_new_params_from_old_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
//                                                      compression_map,num_new_ptr,
//                                                      csum_new_nspix_ptr,nspix_buffer);
//   fill_old_params_from_new_b<<<nblocks1,nthreads1>>>(sp_params,new_params,
//                                                      prev_nspix_ptr,num_new_ptr,
//                                                      csum_new_nspix_ptr,nspix_buffer);

//   // -- free parameters --
//   cudaFree(new_params);
//   cudaFree(compression_map);

// // (spix_params* params, spix_params*  new_params,
// //                                 int* compression_map, int* num_new_nspix, int* csum_new_nspix)
//   return;
// }
