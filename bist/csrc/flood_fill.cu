
#include "flood_fill.h"
#include "atomic_helpers.h"

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

#define THREADS_PER_BLOCK 512



void flood_fill(PointCloudData& data, spix_params* params, uint32_t* spix, uint32_t* csum_nspix, uint32_t S){

    // -- launch parameters --
    int NumThreads = THREADS_PER_BLOCK;
    int vertex_nblocks = ceil( double(data.V) / double(NumThreads) ); 
    dim3 VertexBlocks(vertex_nblocks);

    //
    // Step 1: Keep Only the Closest Point to Center
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
    find_clostest_point<<<VertexBlocks,NumThreads>>>(distances_ptr,data.pos_ptr(),params,spix,csum_nspix,data.vertex_batch_ids_ptr(),data.V);

    // -- keep one point in each spix that is closest --
    set_only_seeds<<<VertexBlocks,NumThreads>>>(distances_ptr,spix,csum_nspix,data.vertex_batch_ids_ptr(),data.V);

    //
    // Step 2: Expansion Loop
    //



    // -- allocate --
    thrust::device_vector<bool> valid_label(data.V, 0);
    bool* valid_label_ptr = thrust::raw_pointer_cast(valid_label.data());
    uint32_t count = thrust::count(valid_label.begin(), valid_label.end(), true);
    while(count < data.V){
        // -- propogate seed value to all connected neighbors --
        propogate_seed<<<VertexBlocks,NumThreads>>>(spix,valid_label_ptr,data.csr_edges_ptr(),data.csr_eptr_ptr(),data.V);
        count = thrust::count(valid_label.begin(), valid_label.end(), true);
        // printf("count: %d\n",count);
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
find_clostest_point(uint64_t* distances, float3* pos, spix_params* params, uint32_t* spix, uint32_t* csum_nspix, uint8_t* bids, uint32_t V){

    // -- unpack indexing --
	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];
    uint32_t spix_id = spix[vertex];
    uint32_t spix_index = spix_id + spix_id_offset;

    // -- compute distance between points --
    double3 mu_pos = params[spix_index].mu_pos;
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
propogate_seed(uint32_t* spix, bool* valid_label, uint32_t* edges, uint32_t* eptr, uint32_t V){

    // -- unpack indexing --
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex >= V) return;
    
    uint32_t my_spix_id = spix[vertex];
    
    // Only labeled vertices participate in propagation
    if (my_spix_id == UINT32_MAX) return;
    valid_label[vertex] = 1;
    
    // -- read neighbors --
    int start = eptr[vertex];
    int end = eptr[vertex+1];
    for(int index = start; index < end; index++){
        uint32_t neigh = edges[index];
        
        // Try to claim this unlabeled neighbor
        uint32_t old_val = atomicCAS(&spix[neigh], UINT32_MAX, my_spix_id);
        // if (old_val == UINT32_MAX){
        //     valid_label[vertex] = 1;
        // }
    }
}