
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

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
);

__global__ void approximate_articulation_points_v0(
    const uint32_t* labels,  // Cluster Labels
    const uint32_t* csr_edges,           // 1-hop neighbor data
    const uint32_t* csr_ptr,             // CSR pointers
    bool* is_simple_point,                // Output: true if simple point
    uint8_t* num_neq, // Output: Num p
    uint32_t V                   // Number of vertices
);