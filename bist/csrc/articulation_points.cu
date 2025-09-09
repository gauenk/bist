/*************************************************************************************************
 * 
 * 
 *    Articulation Points are another name for "NOT" Simple Ponts from the 2D Superpixel Papers 
 *    Check-out: "A Fast Method for Inferring High-Quality Simply-Connected Superpixels"
 *    
 *    In 3D, finding these points is a fundamentally challenging problem. If we have a cluster of
 *    vertices and we are checking if vertex v is simple, we need to check 
 *    if there is ANY path to connect ALL the remaining nodes. This is a global(-ish) condition 
 *    and we only get to check it locally. In 2D, the problem is simple since the dimension is 
 *    only 2D and we know the connectivity between our neighbors.
 * 
 *    We do a 2-hop approximation. For point-clouds lifted from a triangular mesh (ScanNetv2),
 *    I think it's "good enough" for this method..
 * 
 * 
 **************************************************************************************************/


#include "articulation_points.h"

__device__ bool check_2hop_connectivity(
    uint32_t n1, uint32_t n2, uint32_t avoid_vertex,
    const uint32_t* csr_edges, const uint32_t* csr_ptr,
    const uint32_t* neighbors_cache, int num_neighbors_cache
) {
    // 1-hop: Check if n1 and n2 are directly connected
    // Search in the avoid_vertex's neighbor list (cached in shared memory)
    bool n1_found = false, n2_found = false;
    for (int i = 0; i < num_neighbors_cache; i++) {
        if (neighbors_cache[i] == n1) n1_found = true;
        if (neighbors_cache[i] == n2) n2_found = true;
    }
    
    // If both are neighbors of avoid_vertex, check if they're direct neighbors
    if (n1_found && n2_found) {
        // Check global CSR for direct n1-n2 connection
        int start_n1 = csr_ptr[n1];
        int end_n1 = csr_ptr[n1 + 1];
        for (int i = start_n1; i < end_n1; i++) {
            if (csr_edges[i] == n2) return true;
        }
    }
    
    // 2-hop: Check if n1 and n2 share any common neighbors (excluding avoid_vertex)
    int start_n1 = csr_ptr[n1];
    int end_n1 = csr_ptr[n1 + 1];
    
    for (int i = start_n1; i < end_n1; i++) {
        uint32_t common = csr_edges[i];
        if (common == avoid_vertex) continue;
        
        // Check if 'common' is also neighbor of n2
        int start_n2 = csr_ptr[n2];
        int end_n2 = csr_ptr[n2 + 1];
        
        for (int j = start_n2; j < end_n2; j++) {
            if (csr_edges[j] == common) {
                return true;  // Found 2-hop path: n1-common-n2
            }
        }
    }
    
    return false;  // No 2-hop connection found
}

// i -> (row, col), strict upper triangle (excluding diagonal)
__device__ void upper_strict_from_index(uint32_t n, int i, int& row, int& col) {
    int n2 = n - 1;                    // map to an (n-1)x(n-1) upper-inc problem
    int N2 = n2*(n2+1)/2;              // equals n*(n-1)/2
    int j  = N2 - 1 - i;
    int rprime = ( (int)std::floor((std::sqrt(8.0*j+1)-1)/2) );
    int cprime = j - rprime*(rprime+1)/2;
    row = (n2 - 1) - rprime;                 // same row
    col = (n2 - 1) - cprime + 1;             // shift right by 1 to skip the diagonal
}



__global__ void approximate_articulation_points(
    const uint32_t* labels,  // Cluster Labels
    const bool*     border,
    const uint32_t* csr_edges,           // 1-hop neighbor data
    const uint32_t* csr_ptr,             // CSR pointers
    bool* is_simple_point,                // Output: true if simple point
    uint8_t* num_neq, // Output: Num p
    uint8_t* gcolors,
    uint8_t gchrome,
    uint32_t V                   // Number of vertices
) {
    

    // Warp and thread identification
    uint32_t vertex = (blockIdx.x * blockDim.x + threadIdx.x);
    if (vertex >= V) return;

    uint32_t my_label = labels[vertex];
    uint8_t gcolor = gcolors[vertex];
    if (gcolor != gchrome){ return; }
    if (!border[vertex]) { return; }

    uint32_t start = csr_ptr[vertex];
    uint32_t end = csr_ptr[vertex+1];
    // if ((end - start) > 3){
    //     printf("vertex[%d]: # of edges %d %d\n",vertex,start,end);
    // }
    assert( (end - start) <= 3);
    //bool any_neq = false;
    uint8_t num_neq_v = 0;
    for(int index = start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        uint32_t neigh_vertex = labels[neigh];
        num_neq_v += neigh_vertex != my_label;
    }

    num_neq[vertex] = num_neq_v;
    is_simple_point[vertex] = num_neq_v == 1;
    
 
}


__global__ void approximate_articulation_points_v0(
    const uint32_t* labels,  // Cluster Labels
    const uint32_t* csr_edges,           // 1-hop neighbor data
    const uint32_t* csr_ptr,             // CSR pointers
    bool* is_simple_point,                // Output: true if simple point
    uint8_t* num_neq, // Output: Num p
    uint32_t V                   // Number of vertices
) {
    
    //  Warp Sizing
    constexpr uint32_t NUM_WARPS = 8;
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint8_t MAX_NEIGH = 32; // just happens to be 32; could be anything; 32 happens to also be # threads in warp.

    // Warp and thread identification
    uint32_t global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;        // 0-7 (8 warps per block)
    int lane_id = threadIdx.x % WARP_SIZE;
    assert(warp_in_block < NUM_WARPS);
    
    if (global_warp_id >= V) return;
    
    //uint32_t vertex = candidates[global_warp_id];
    uint32_t vertex = global_warp_id;
    uint32_t my_label = labels[vertex];
    
    // Shared memory: 8 warps × 32 neighbors = 1KB per block
    __shared__ uint32_t shared_neighbors[NUM_WARPS][MAX_NEIGH];
    __shared__ uint32_t shared_labels[NUM_WARPS][MAX_NEIGH];
    __shared__ int shared_num_neighbors[NUM_WARPS];      // One per warp
    
    // Step 1: Cooperatively load 1-hop neighbors (coalesced)
    int start = csr_ptr[vertex];
    uint32_t num_neighbors = csr_ptr[vertex + 1] - start;
    shared_num_neighbors[warp_in_block] = num_neighbors;
    
    // Read Read Read
    for (int offset = 0; offset < MAX_NEIGH; offset += WARP_SIZE) {
        int idx = offset + lane_id;
        if (idx < num_neighbors && idx < MAX_NEIGH) {
            uint32_t neighbor = csr_edges[start + idx];
            shared_neighbors[warp_in_block][idx] = neighbor;
            shared_labels[warp_in_block][idx] = labels[neighbor];  // Direct access
        }
    }

    __syncwarp();

    // Count neighbors with different cluster labels
    int num_different_label = 0;
    for (int i = 0; i < num_neighbors && i < MAX_NEIGH; i++) {
        if (shared_labels[warp_in_block][i] != my_label) {
            num_different_label++;
        }
    }

    // Store result using lane 0
    if (lane_id == 0) {
        num_neq[global_warp_id] = num_different_label;
    }
    
    // Early exit for trivial cases
    if (num_neighbors <= 1) {
        if (lane_id == 0) {
            is_simple_point[global_warp_id] = true;  // Definitely simple
        }
        return;
    }
    
    // Handle vertices with too many neighbors
    if (num_neighbors >= MAX_NEIGH) {
        if (lane_id == 0) {
            is_simple_point[global_warp_id] = false;  // Conservative: assume not simple
        }
        return;
    }
    
    // Step 2: Check all neighbor pairs for 2-hop connectivity
    int total_pairs = (num_neighbors * (num_neighbors - 1)) / 2;
    bool thread_found_disconnection = false;
    
    // Each thread handles subset of pairs
    int pair_id = lane_id;
    while (pair_id < total_pairs && !thread_found_disconnection) {
        // Convert linear pair_id to (i,j) indices where i < j
        int temp_pair = pair_id;
        
        // Go from (pair_id) -> (i,j) neighbors
        // int i = (int)((sqrtf(1 + 8.0f * pair_id) - 1) / 2);
        // int j = pair_id - (i * (i + 1)) / 2 + i + 1;
        int i,j;
        upper_strict_from_index(num_neighbors, pair_id, i, j);
        
        if (i < num_neighbors && j < num_neighbors) {
            uint32_t n1 = shared_neighbors[warp_in_block][i];
            uint32_t n2 = shared_neighbors[warp_in_block][j];
            uint32_t l1 = shared_labels[warp_in_block][i];
            uint32_t l2 = shared_labels[warp_in_block][j];

            // Skip if either label isn't in the same cluster;
            // if ((l1 != my_label) || (l2 != my_label)){ continue; }

            // Check 2-hop connectivity
            bool connected;
            if ((l1 == my_label) && (l2 == my_label)){
                connected = check_2hop_connectivity(
                    n1, n2, vertex, 
                    csr_edges, csr_ptr,
                    shared_neighbors[warp_in_block], 
                    num_neighbors);
            }
            // }else{
            //     connected = false;
            // }
            
            if (!connected) {
                thread_found_disconnection = true;
            }
        }
        
        pair_id += WARP_SIZE;  // Next pair for this thread
    }
    
    // Step 3: Warp-level reduction
    bool warp_has_disconnection = __any_sync(0xFFFFFFFF, thread_found_disconnection);
    
    // Write result (simple point if no disconnections found)
    if (lane_id == 0) {
        is_simple_point[global_warp_id] = !warp_has_disconnection;
    }
}
