

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/count.h>

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
//     int num_vertices
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_vertices) return;
    
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
//     int num_vertices
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_vertices) return;
    
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
//     int num_vertices
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_vertices) return;
    
//     int vertex = tid;
    
//     if (selected[vertex]) {
//         colors[vertex] = current_color;
//         uncolored[vertex] = false;
//     }
// }

// Initialize random number generators
__global__ void init_curand_kernel(
    curandState* states,
    unsigned long seed,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// Generate priorities for uncolored vertices
__global__ void fuzed_kernel(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,      // CSR edge pointers (size V+1)
    uint8_t* colors,           // Output color assignments
    float* priorities,         // Output priorities
    bool* uncolored,     // Which vertices are still uncolored
    curandState* states,       // Random number generators
    int num_vertices,
    uint8_t current_color
) {

    // -- init --
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    int vertex = tid;
    
    //
    //
    //  Step 1: Compute my Priority
    //
    //

    if (!uncolored[vertex]) {
        priorities[vertex] = -1.0f; // Invalid priority for colored vertices
        return;
    }
    
    // Priority = degree + random number
    int start = eptr[vertex];
    int end = eptr[vertex + 1];
    int degree = end - start;
    float random_val = curand_uniform(&states[vertex]);
    float my_priority = (float)degree + random_val;
    priorities[vertex] = my_priority;

    // -- global synchronization --
    grid_group g = this_grid();
    g.sync();

    //
    //
    // Step 2: Select Vertices to Color Based on Priority
    //
    //

    // Check all neighbors   
    bool is_local_maximum = true; 
    for (int i = start; i < end; i++) {
        int neighbor = edges[i];
        
        // Only consider uncolored neighbors
        if (uncolored[neighbor]) {
            float neighbor_priority = priorities[neighbor];
            
            // If neighbor has higher or equal priority, we're not the local max
            if (neighbor_priority >= my_priority) {
                is_local_maximum = false;
                break;
            }
        }
    }
    
    //
    //
    // Step 3: Color the Proper Vertices
    //
    //

    if (is_local_maximum){
        colors[vertex] = current_color;
        uncolored[vertex] = false;
    }

}


// Main Luby's coloring function
uint8_t* get_graph_coloring(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,       // CSR edge pointers (size V+1)
    int num_vertices) {

    // -- allocate --
    uint8_t* colors = (uint8_t*)easy_allocate(num_vertices,sizeof(uint8_t));
    const int block_size = 256;
    const int grid_size = (num_vertices + block_size - 1) / block_size;
    
    // Initialize device arrays
    thrust::device_vector<float> d_priorities(num_vertices);
    thrust::device_vector<bool> d_uncolored(num_vertices, true);
    thrust::device_vector<curandState> d_curand_states(num_vertices);
    
    // Initialize random number generators
    init_curand_kernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_curand_states.data()),
        time(NULL),
        num_vertices
    );
    cudaDeviceSynchronize();

    // -- pointers --
    float* d_prior_ptr = thrust::raw_pointer_cast(d_priorities.data());
    bool* d_uncolored_ptr = thrust::raw_pointer_cast(d_uncolored.data());
    curandState* d_curand_ptr = thrust::raw_pointer_cast(d_curand_states.data());

    // Main Luby's algorithm loop
    uint8_t current_color = 0;
    void* args[] = {
        (void*)&edges,
        (void*)&eptr,
        (void*)&colors,
        (void*)&d_prior_ptr,
        (void*)&d_uncolored_ptr,
        (void*)&d_curand_ptr,
        (void*)&num_vertices,
        (void*)&current_color
    };
    const int max_colors = 250; // Safety limit
    while (current_color < max_colors) {
        // Check if any vertices remain uncolored
        int uncolored_count = thrust::count(d_uncolored.begin(), d_uncolored.end(), true);
        if (uncolored_count == 0) break;
        // void* args[] = {edges,eptr,colors,
        //     thrust::raw_pointer_cast(d_priorities.data()),
        //     thrust::raw_pointer_cast(d_uncolored.data()),
        //     thrust::raw_pointer_cast(d_curand_states.data()),
        //     num_vertices,current_color};
        cudaLaunchCooperativeKernel((void*)fuzed_kernel, grid_size, block_size, args);     
        cudaDeviceSynchronize();
        current_color++;

        // // Step 1: Generate priorities for uncolored vertices
        // generate_priorities_kernel<<<grid_size, block_size>>>(
        //     eptr,
        //     thrust::raw_pointer_cast(d_priorities.data()),
        //     thrust::raw_pointer_cast(d_uncolored.data()),
        //     thrust::raw_pointer_cast(d_curand_states.data()),
        //     num_vertices
        // );
        // cudaDeviceSynchronize();
        
        // // Step 2: Find independent set
        // find_independent_set_kernel<<<grid_size, block_size>>>(
        //     csr_edges,eptr,
        //     thrust::raw_pointer_cast(d_priorities.data()),
        //     thrust::raw_pointer_cast(d_uncolored.data()),
        //     thrust::raw_pointer_cast(d_selected.data()),
        //     num_vertices
        // );
        // cudaDeviceSynchronize();
        
        // // Step 3: Color selected vertices
        // color_vertices_kernel<<<grid_size, block_size>>>(
        //     thrust::raw_pointer_cast(d_selected.data()),
        //     thrust::raw_pointer_cast(d_colors.data()),
        //     thrust::raw_pointer_cast(d_uncolored.data()),
        //     current_color,
        //     num_vertices
        // );

    }

    return colors;
}
