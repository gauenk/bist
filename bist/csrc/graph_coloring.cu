

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "init_utils.h"
#include "graph_coloring.h"


// // Generate priorities for uncolored vertices
// __global__ void generate_priorities_kernel(
//     const uint32_t* eptr,      // CSR edge pointers (size V+1)
//     float* priorities,         // Output priorities
//     const bool* uncolored,     // Which vertices are still uncolored
//     curandState* states,       // Random number generators
//     int V
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= V) return;
    
//     int vertex = tid;
    
//     if (!uncolored[vertex]) {
//         priorities[vertex] = -1.0f; // Invalid priority for colored vertices
//         return;
//     }
    
//     // Priority = degree + random number
//     int degree = eptr[vertex + 1] - eptr[vertex];
//     float random_val = curand_uniform(&states[vertex]);
//     priorities[vertex] = (float)degree + random_val;
// }

// // Find independent set using Luby's algorithm
// __global__ void find_independent_set_kernel(
//     const uint32_t* csr_edges,  // CSR edge list
//     const uint32_t* eptr,       // CSR edge pointers (size V+1)
//     const float* priorities,    // Vertex priorities
//     const bool* uncolored,      // Which vertices are uncolored
//     bool* selected,             // Output: vertices selected for independent set
//     int V
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= V) return;
    
//     int vertex = tid;
    
//     // Skip colored vertices
//     if (!uncolored[vertex]) {
//         selected[vertex] = false;
//         return;
//     }
    
//     float my_priority = priorities[vertex];
//     bool is_local_maximum = true;
    
//     // Check all neighbors
//     int start = eptr[vertex];
//     int end = eptr[vertex + 1];
    
//     for (int i = start; i < end; i++) {
//         int neighbor = csr_edges[i];
        
//         // Only consider uncolored neighbors
//         if (uncolored[neighbor]) {
//             float neighbor_priority = priorities[neighbor];
            
//             // If neighbor has higher or equal priority, we're not the local max
//             if (neighbor_priority >= my_priority) {
//                 is_local_maximum = false;
//                 break;
//             }
//         }
//     }
    
//     selected[vertex] = is_local_maximum;
// }


// // Color selected vertices and mark them as colored
// __global__ void color_vertices_kernel(
//     const bool* selected,       // Which vertices to color
//     uint8_t* colors,           // Output color assignments
//     bool* uncolored,           // Mark these vertices as colored
//     uint8_t current_color,
//     int V
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= V) return;
    
//     int vertex = tid;
    
//     if (selected[vertex]) {
//         colors[vertex] = current_color;
//         uncolored[vertex] = false;
//     }
// }


__device__ float hash_random(unsigned int seed, unsigned int tid) {
    unsigned int hash = tid;
    hash ^= seed;
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return (float)(hash & 0xFFFFFF) / (float)0xFFFFFF;
}


// Generate priorities for uncolored vertices
template<uint8_t _V_PER_THREAD>
__global__ void kernel_step_1(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,      // CSR edge pointers (size V+1)
    uint8_t* colors,           // Output color assignments
    float* priorities,         // Output priorities
    bool* uncolored,     // Which vertices are still uncolored
    int V,
    uint8_t current_color,
    unsigned long random_seed
) {

    // -- init --
    //const int V_PER_THREAD=4;
    constexpr uint8_t V_PER_THREAD = _V_PER_THREAD;  // Convert to constexpr
    int tid = V_PER_THREAD*(blockIdx.x * blockDim.x + threadIdx.x);
    int Vmax = tid + V_PER_THREAD;
    float local_priorities[V_PER_THREAD];
    uint32_t local_eptr[V_PER_THREAD+1];
    int _ix;

    //
    //
    //  Step 1: Compute my Priority
    //
    //

    // Step 1a: Reading all at one... maybe better?
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex <= Vmax; ++vertex){
        if (vertex > V){
            _ix++;
            continue;
        }
        local_eptr[_ix] = eptr[vertex];
        _ix+=1;
    }

    // Step 1b: Compute Priorities
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        if (!uncolored[vertex]) {
            local_priorities[_ix] = -1.0f; // Invalid priority for colored vertices
            _ix++;
            continue;
        }
        
        // Priority = degree + random number
        // int start = local_eptr[_ix];
        // int end = local_eptr[_ix + 1];
        int degree = local_eptr[_ix + 1] - local_eptr[_ix];
        float random_val = hash_random(random_seed, vertex);
        float my_priority = (float)degree + random_val;
        //priorities[vertex] = my_priority;
        local_priorities[_ix] = my_priority;
        _ix+=1;

    }
    
    // Step 1c: Write All at Once
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        priorities[vertex] = local_priorities[_ix];
        _ix+=1;
    }

}

// Generate priorities for uncolored vertices
template<uint8_t _V_PER_THREAD>
__global__ void kernel_step_2(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,      // CSR edge pointers (size V+1)
    uint8_t* colors,           // Output color assignments
    float* priorities,         // Output priorities
    bool* uncolored,     // Which vertices are still uncolored
    bool* local_max,
    int V,
    uint8_t current_color,
    unsigned long random_seed
) {

    // -- init --
    //const int V_PER_THREAD=4;
    constexpr uint8_t V_PER_THREAD = _V_PER_THREAD;  // Convert to constexpr
    int tid = V_PER_THREAD*(blockIdx.x * blockDim.x + threadIdx.x);
    int Vmax = tid + V_PER_THREAD;
    float local_priorities[V_PER_THREAD];
    uint32_t local_eptr[V_PER_THREAD+1];
    bool is_local_maximum[V_PER_THREAD];
    bool local_uncolored[V_PER_THREAD];
    int _ix;

    // if (tid == (V-1)){
    //     printf("info [%d,%d,%d]\n",tid,V,Vmax);
    // }


    //
    //
    // Step 2: Select Vertices to Color Based on Priority
    //
    //

    // Step 2a: Reading all at one... maybe better?
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex <= Vmax; ++vertex){
        if (vertex > V){
            _ix++;
            continue;
        }
        local_eptr[_ix] = eptr[vertex];
        _ix+=1;
    }
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        local_priorities[_ix] = priorities[vertex];
        _ix+=1;
    }
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        local_uncolored[_ix] = uncolored[vertex];
        _ix+=1;
    }

    // Step 2b: Compute Bool for Local Maximum
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        // int vertex = tid + _ix;
        if (vertex >= V){
            _ix++;
            continue;
        }

        // Only check if vertex is uncolored
        if (!local_uncolored[_ix]) {
            is_local_maximum[_ix] = false;
            _ix++;
            continue;
        }

        float my_priority = local_priorities[_ix];
        // Check all neighbors   
        is_local_maximum[_ix] = true; 
        for (uint32_t i = local_eptr[_ix]; i < local_eptr[_ix+1]; i++) {
            int neighbor = edges[i];
    

            // Only consider uncolored neighbors
            if (uncolored[neighbor]) {
                float neighbor_priority = priorities[neighbor];
                
                // // If neighbor has higher or equal priority, we're not the local max
                // if (neighbor_priority >= my_priority) {
                //     is_local_maximum[_ix] = false;
                //     break;
                // }

                if (neighbor_priority > my_priority || 
                    (neighbor_priority == my_priority && neighbor > vertex)) {
                    is_local_maximum[_ix] = false;
                    break;
                }
            }
        }
        _ix++;
    }

    // -- write --
    _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        local_max[vertex] = is_local_maximum[_ix];
    }
    _ix++;

}

// Generate priorities for uncolored vertices
template<uint8_t _V_PER_THREAD>
__global__ void kernel_step_3(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,      // CSR edge pointers (size V+1)
    uint8_t* colors,           // Output color assignments
    float* priorities,         // Output priorities
    bool* uncolored,     // Which vertices are still uncolored
    bool* local_max,
    int V,
    uint8_t current_color,
    unsigned long random_seed
) {

    // -- init --
    //const int V_PER_THREAD=4;
    constexpr uint8_t V_PER_THREAD = _V_PER_THREAD;  // Convert to constexpr
    int tid = V_PER_THREAD*(blockIdx.x * blockDim.x + threadIdx.x);
    int Vmax = tid + V_PER_THREAD;
    // float local_priorities[V_PER_THREAD];
    // int local_eptr[V_PER_THREAD+1];
    // bool is_local_maximum[V_PER_THREAD];
    // bool local_uncolored[V_PER_THREAD];
    // int _ix;

    //
    //
    // Step 3: Color the Proper Vertices
    //
    //

    int _ix = 0;
    //#pragma unroll V_PER_THREAD
    for (int vertex=tid; vertex < Vmax; ++vertex){
        if (vertex >= V){
            _ix++;
            continue;
        }
        if (local_max[vertex]){
            colors[vertex] = current_color;
            uncolored[vertex] = false;
        }
        _ix++;
    }

}

// template<uint8_t MAX_COLORS, uint8_t _V_PER_THREAD>
// __global__ void recolor_greedy(
//     const uint32_t* edges,
//     const uint32_t* eptr,
//     const uint8_t* old_colors,    // Read from here
//     uint8_t* new_colors,          // Write to here
//     int V
// ) {
//     // -- init --
//     constexpr uint8_t V_PER_THREAD = _V_PER_THREAD;  // Convert to constexpr
//     int tid = V_PER_THREAD*(blockIdx.x * blockDim.x + threadIdx.x);
//     int Vmax = tid + V_PER_THREAD;
//     uint32_t local_eptr[V_PER_THREAD+1];
//     bool used_colors[MAX_COLORS] = {false};
//     int _ix;

//     // -- read --
//     _ix = 0;
//     //#pragma unroll V_PER_THREAD
//     for (int vertex=tid; vertex <= Vmax; ++vertex ){
//         if (vertex > V){
//             _ix++;
//             continue;
//         }
//         local_eptr[_ix] = eptr[vertex];
//         _ix++;
//     }

//     // -- recolor --
//     _ix = 0;
//     //#pragma unroll V_PER_THREAD
//     for (int vertex=tid; vertex < Vmax; ++vertex ){
//         if (vertex >= V){
//             _ix++;
//             continue;
//         }

//         //#pragma unroll MAX_COLORS
//         for (int c = 0; c < MAX_COLORS; c++) { 
//             used_colors[c] = false;
//         }
        
//         // Mark colors used by neighbors (reading from old_colors)
//         for (uint32_t i = local_eptr[_ix]; i < local_eptr[_ix + 1]; i++) {
//             int neighbor = edges[i];
//             if (neighbor != vertex) {
//                 used_colors[old_colors[neighbor]] = true;
//             }
//         }
        
//         // Find lowest available color and write to new_colors
//         //#pragma unroll MAX_COLORS  // or whatever your max possible colors is
//         for (uint8_t c = 0; c < MAX_COLORS; c++) {
//             if (!used_colors[c]) {
//                 new_colors[vertex] = c;
//                 break;
//             }
//         }
//         //new_colors[vertex] = 0;
//         _ix++;
//     }
// }

__global__ void validate_coloring(
    const uint32_t* edges,
    const uint32_t* eptr,
    const uint8_t* colors,
    bool* is_valid,  // Output: per-vertex validity
    int V
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    
    uint8_t my_color = colors[vertex];
    bool valid = true;
    
    // Check all neighbors
    for (int i = eptr[vertex]; i < eptr[vertex + 1]; i++) {
        int neighbor = edges[i];
        if (neighbor != vertex && colors[neighbor] == my_color) {
            valid = false;
            //printf("vertex: %d, %d, %d\n",vertex,neighbor,my_color);
            break;
        }
    }
    
    is_valid[vertex] = valid;
}

// Main Luby's coloring function
std::tuple<uint8_t*,uint8_t> 
get_graph_coloring(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,       // CSR edge pointers (size V+1)
    int V) {

    //check_cooperative_limits();

    // -- consts --
    const uint8_t V_PER_THREAD = 1;
    const uint8_t MAX_COLORS = 32; // Safety limit

    // -- allocate --
    uint8_t* colors = (uint8_t*)easy_allocate(V,sizeof(uint8_t));
    cudaMemset(colors,0x01,V*sizeof(uint8_t));
    const int block_size = 512;
    const int grid_size = (V/V_PER_THREAD + block_size - 1) / block_size;
    
    // Initialize device arrays
    thrust::device_vector<float> d_priorities(V);
    thrust::device_vector<bool> d_uncolored(V, true);
    thrust::device_vector<bool> d_is_local_max(V, false);

    // -- pointers --
    float* d_prior_ptr = thrust::raw_pointer_cast(d_priorities.data());
    bool* d_uncolored_ptr = thrust::raw_pointer_cast(d_uncolored.data());
    bool* d_local_max_ptr = thrust::raw_pointer_cast(d_is_local_max.data());
    
    // Main Luby's algorithm loop
    uint8_t current_color = 0;
    unsigned long random_seed = 123;
    while (current_color < MAX_COLORS) {
        // Check if any vertices remain uncolored
        int uncolored_count = thrust::count(d_uncolored.begin(), d_uncolored.end(), true);
        if (uncolored_count == 0) break;
        kernel_step_1<V_PER_THREAD><<<grid_size, block_size>>>(edges,eptr,colors,d_prior_ptr,
                                                  d_uncolored_ptr,V,current_color,random_seed);     
        cudaDeviceSynchronize();
        kernel_step_2<V_PER_THREAD><<<grid_size, block_size>>>(edges,eptr,colors,d_prior_ptr,
                                                                d_uncolored_ptr,d_local_max_ptr,V,current_color,random_seed);  
        cudaDeviceSynchronize();
        kernel_step_3<V_PER_THREAD><<<grid_size, block_size>>>(edges,eptr,colors,d_prior_ptr,
                                                              d_uncolored_ptr,d_local_max_ptr,V,current_color,random_seed);  
        cudaDeviceSynchronize();
        current_color++;
        random_seed = random_seed * 1103515245 + 12345; // Simple LCG update
    }
    if (current_color >= MAX_COLORS){
        printf("Something bad happened when coloring your graph :/\n");
        exit(1);
    }

    // -- try to reduce chromatic number [bad attempt; race condition] --
    // uint8_t* new_colors = (uint8_t*)easy_allocate(V,sizeof(uint8_t));
    // cudaDeviceSynchronize();
    // recolor_greedy<MAX_COLORS,V_PER_THREAD><<<grid_size, block_size>>>(edges,eptr,colors,new_colors,V);
    // auto dptr = thrust::device_ptr<uint8_t>(new_colors);
    // cudaDeviceSynchronize();
    // uint8_t new_chromaticity = *thrust::max_element(dptr,dptr + V) + 1; // +1 since colors are 0-indexed
    // //printf("[chroma] old,new (%d,%d)\n",current_color,new_chromaticity);

    // -- check --
    const int _block_size = 512;
    const int _grid_size = (V + block_size - 1) / block_size;
    bool* is_valid = (bool*)easy_allocate(V,sizeof(bool));
    validate_coloring<<<_grid_size, _block_size>>>(edges,eptr,colors,is_valid,V);
    thrust::device_vector<bool> valid_vec(is_valid, is_valid + V);
    int valid_count = thrust::count(valid_vec.begin(), valid_vec.end(), true);
    bool all_valid = (valid_count == V);
    if (all_valid) {
        printf("✓ Graph coloring is valid!\n");
    } else {
        printf("✗ Found %d vertices with invalid coloring\n", V - valid_count);
    }
    cudaFree(is_valid);


    return std::tuple(colors,current_color);

}
