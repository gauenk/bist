/*

    Using flood fill only for the initial cluster ids

*/



#include "flood_fill.h"
#include "atomic_helpers.h"


#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

#define THREADS_PER_BLOCK 512



void flood_fill(PointCloudData& data, uint32_t* spix, uint32_t* nspix, uint32_t* csum_nspix, uint32_t S){

    // -- launch parameters --
    int NumThreads = THREADS_PER_BLOCK;
    int VertexBlocks = ceil( double(data.V) / double(NumThreads) ); 
    int SpixBlocks = ceil( double(S) / double(NumThreads) );

    //
    // Step 1: Compute Spatial Mean
    //

    thrust::device_vector<float3> mu_pos(S);
    float3* mu_pos_dptr = thrust::raw_pointer_cast(mu_pos.data());
    cudaMemset(mu_pos_dptr,0,S*sizeof(float3));
    thrust::device_vector<uint32_t> scounts(S,0);
    uint32_t* scounts_dptr = thrust::raw_pointer_cast(scounts.data());
    reduce_pos<<<VertexBlocks,NumThreads>>>(mu_pos_dptr, scounts_dptr, data.pos_ptr(), spix, csum_nspix, data.vertex_batch_ids_ptr(), data.V, S);
    normalize_pos<<<SpixBlocks,NumThreads>>>(mu_pos_dptr, scounts_dptr, S);

    
    //
    // Step 2: Keep Only the Closest Point to Center
    //

    // -- allocate --
    thrust::device_vector<uint64_t> distances(S, 0);
    uint64_t* distances_ptr = thrust::raw_pointer_cast(distances.data());

    // -- init --
    float init_dist = 1e9f;  // Large distance
    uint32_t init_vertex_id = UINT32_MAX;  // Invalid vertex ID
    uint32_t init_dist_bits = *reinterpret_cast<uint32_t*>(&init_dist);
    uint64_t init_packed = (uint64_t(init_dist_bits) << 32) | uint64_t(init_vertex_id);
    thrust::fill(distances.begin(), distances.end(), init_packed);

    // -- find min sizes --
    find_clostest_point<<<VertexBlocks,NumThreads>>>(distances_ptr,data.pos_ptr(),mu_pos_dptr,spix,csum_nspix,data.vertex_batch_ids_ptr(),data.V);

    // -- keep one point in each spix that is closest --
    set_only_seeds<<<VertexBlocks,NumThreads>>>(distances_ptr,spix,csum_nspix,data.vertex_batch_ids_ptr(),data.V);

    //
    // Step 3: Expansion Loop
    //

    // -- propogate min-label to identify unlabeled connected components --
    thrust::device_vector<uint32_t> mlabels(data.V,UINT32_MAX);
    uint32_t* mlabels_dptr = thrust::raw_pointer_cast(mlabels.data());

    // -- allocate --
    thrust::device_vector<bool> valid_label(data.V, 0);
    bool* valid_label_ptr = thrust::raw_pointer_cast(valid_label.data());
    uint32_t count = thrust::count(valid_label.begin(), valid_label.end(), true);
    int iter = 0;
    int count_prev = -1;
    while(count < data.V){

        // -- propogate seed value to all connected neighbors --
        propogate_seed<<<VertexBlocks,NumThreads>>>(spix,valid_label_ptr,mlabels_dptr,data.csr_edges_ptr(),data.csr_eptr_ptr(),data.V,iter);
        count = thrust::count(valid_label.begin(), valid_label.end(), true);
        if ((count == count_prev) && (count < data.V)){ // assume connected components have been propogated -- might oversegment a bit but we don't care

            // -- map "min_vertex" to compacted ids --
            thrust::device_vector<uint32_t> imap(data.V,UINT32_MAX);
            uint32_t* imap_dptr = thrust::raw_pointer_cast(imap.data());
            thrust::device_vector<uint32_t> num_new_spix(data.B,0); // writer offset
            uint32_t* num_new_spix_dptr = thrust::raw_pointer_cast(num_new_spix.data());
            set_imap<<<VertexBlocks,NumThreads>>>(imap_dptr,mlabels_dptr,spix,num_new_spix_dptr,data.vertex_batch_ids_ptr(),data.V);

            // -- set connected components via mininimum vertex id if invalid --
            set_invalid_components<<<VertexBlocks,NumThreads>>>(imap_dptr,mlabels_dptr,spix,data.V);

            // -- update number of superpixels --
            add_new_nspix<<<1,data.B>>>(nspix,num_new_spix_dptr,data.B);

            break;
            // printf("count: %d %d\n",count,data.V);
            // printf("Something is wrong with your mesh -- its not path-connected.\n");
            // if (iter > 1){
            //     exit(1);
            // }
            // iter++;
        }
        count_prev = count;
    }

    // -- view --
    // thrust::host_vector<uint32_t> spix_cpu(data.V);
    // thrust::device_ptr<uint32_t> spix_dev(spix);
    // thrust::copy(spix_dev, spix_dev + data.V, spix_cpu.begin());
    // for(int ix = 0; ix < 10; ix++){
    //     printf("spix[%d] = %d\n",ix,spix_cpu[ix]);
    // }
    // for(int ix = data.V-10; ix < data.V; ix++){
    //     printf("spix[%d] = %d\n",ix,spix_cpu[ix]);
    // }

}

// Helper functions for unpacking
// __device__ float extract_distance(uint64_t packed) {
//     uint32_t dist_bits = uint32_t(packed >> 32);
//     return *reinterpret_cast<float*>(&dist_bits);
// }

// __device__ uint32_t extract_id(uint64_t packed) {
//     return uint32_t(packed & 0xFFFFFFFF);
// }



__global__ void
reduce_pos(float3* mu_pos_vec, uint32_t* count, float3* pos_vec, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V, uint32_t S){

    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;

    // -- batching info --
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];
    uint32_t spix_id_raw = spix[vertex];
    uint32_t spix_id = spix_id_raw + spix_id_offset;
    assert(spix_id < S);

    // -- read --
    float3 pos = pos_vec[vertex];

    // -- accumulate --
    atomicAdd(&mu_pos_vec[spix_id].x,pos.x);
    atomicAdd(&mu_pos_vec[spix_id].y,pos.y);
    atomicAdd(&mu_pos_vec[spix_id].z,pos.z);
    atomicAdd(&count[spix_id],1);
}

__global__ void
normalize_pos(float3* mu_pos, uint32_t* count, uint32_t S){

    // -- unpack indexing --
	uint32_t spix = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (spix>=S) return;

    // -- normalize --
    mu_pos[spix].x = mu_pos[spix].x / count[spix];
    mu_pos[spix].y = mu_pos[spix].y / count[spix];
    mu_pos[spix].z = mu_pos[spix].z / count[spix];

}

__global__ void
find_clostest_point(uint64_t* distances, float3* pos, float3* mu_pos_v, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V){

    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];
    uint32_t spix_id = spix[vertex];
    uint32_t spix_index = spix_id + spix_id_offset;

    // -- compute distance between points --
    float3 mu_pos = mu_pos_v[spix_index];
    float3 v_pos = pos[vertex];
    float delta = (mu_pos.x - v_pos.x)*(mu_pos.x - v_pos.x) + \
                  (mu_pos.y - v_pos.y)*(mu_pos.y - v_pos.y) + \
                  (mu_pos.z - v_pos.z)*(mu_pos.z - v_pos.z); 

    // -- 
    atomic_min_update_float_uint32(distances + spix_index, delta, vertex);
}

__global__ void
set_only_seeds(uint64_t* distances, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V){

    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];
    uint32_t spix_id = spix[vertex];
    uint32_t spix_index = spix_id + spix_id_offset;

    // -- read off selected vertex --
    uint32_t min_vertex = extract_id(distances[spix_index]);
    if (vertex != min_vertex){
        spix[vertex] = UINT32_MAX;
    }
}

__global__ void
propogate_seed(uint32_t* spix, bool* valid_label, uint32_t* mlabels, uint32_t* edges, uint32_t* eptr, uint32_t V, int dev_iter){

    // -- unpack indexing --
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex >= V) return;

    uint32_t my_min = atomicMin(&mlabels[vertex],vertex);
    uint32_t my_spix_id = spix[vertex];
    bool my_id_is_valid = my_spix_id != UINT32_MAX;

    // -- read neighbors --
    int start = eptr[vertex];
    int end = eptr[vertex+1];
    for(int index = start; index < end; index++){
        uint32_t neigh = edges[index];
        
        // -- broadcast your min --
        atomicMin(&mlabels[neigh],my_min);

        // Try to claim this unlabeled neighbor
        if(my_id_is_valid){
            atomicCAS(&spix[neigh], UINT32_MAX, my_spix_id);
        }

    }

    valid_label[vertex] = my_id_is_valid;

}


// __global__ void
// set_is_a_min(bool* is_a_min,  uint32_t* mlabels, uint32_t* spix, uint32_t V){

//     // -- unpack indexing --
//     uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
//     if (vertex >= V) return;

//     // -- skip if spix-id is valid --
//     uint32_t my_spix_id = spix[vertex];
//     bool my_id_is_valid = my_spix_id != UINT32_MAX;
//     if (my_id_is_valid){ return; }
    
//     // -- otherwise, set via label --
//     uint32_t my_min = mlabels[vertex];
//     assert(my_min < UINT32_MAX);
//     is_a_min[my_min] = true;
// }





__global__ void
set_imap(uint32_t* imap, uint32_t* mlabels, uint32_t* spix, uint32_t* nspix_new, uint8_t* bids, uint32_t V){

    /*
     
     to create this imap, we have "set_imap"...
     if invalid, every races to write at "min_vertex" the value "0"
     the winner, now only one, can incriment the "counter" to get an index and then write this value back at min_vertex. 
     
    */

    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;

    // -- skip if spix-id is valid --
    uint32_t my_spix_id = spix[vertex];
    bool my_id_is_valid = my_spix_id != UINT32_MAX;
    if (my_id_is_valid){ return; }

    // -- race for write access --
    uint32_t min_vertex = mlabels[vertex];
    uint32_t read = atomicCAS(&imap[min_vertex],UINT32_MAX,0);
    if (read != UINT32_MAX){ return; } // lost the race

    // -- get offset for the new id --
    uint32_t bx = bids[vertex];
    uint32_t new_spix_id = atomicAdd(&nspix_new[bx],1);

    // -- write (always one thread anyway) --
    imap[vertex] = new_spix_id;
}



// imap_dptr,spix,num_new_spix_dptr,nspix,data.vertex_batch_ids_ptr(),data.V);
__global__ void
set_invalid_components(uint32_t* imap, uint32_t* mlabels, uint32_t* spix, uint32_t V){

    
    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;

    // -- skip if spix-id is valid --
    uint32_t my_spix_id = spix[vertex];
    bool my_id_is_valid = my_spix_id != UINT32_MAX;
    if (my_id_is_valid){ return; }

    // -- set to valid label --
    uint32_t min_vertex = mlabels[vertex];
    spix[vertex] = imap[min_vertex];

}

__global__ void add_new_nspix(uint32_t* nspix, uint32_t* nspix_new, uint32_t B) {
    // -- unpack indexing --
	uint32_t batch_ix = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (batch_ix>=B) return;
    nspix[batch_ix] += nspix_new[batch_ix];
}