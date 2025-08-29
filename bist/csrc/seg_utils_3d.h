#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>

// #include "structs_3d.h"


__host__ void 
set_border(const uint32_t* spix, bool* border,
            const uint32_t* csr_edges, // 1-hop neighbor data
            const uint32_t* csr_ptr,  // CSR pointers
            const uint32_t V);

__host__ void 
set_border_end(const uint32_t* spix, bool* border,
                const uint32_t* csr_edges, // 1-hop neighbor data
                const uint32_t* csr_ptr,  // CSR pointers
                const uint32_t V);

__global__ void 
find_border_vertex(const uint32_t* spix, bool* border,
                    const uint32_t* csr_edges, // 1-hop neighbor data
                    const uint32_t* csr_ptr,  // CSR pointers
                    const uint32_t V);

__global__ void 
find_border_vertex_end(const uint32_t* spix, bool* border,
                        const uint32_t* csr_edges, // 1-hop neighbor data
                        const uint32_t* csr_ptr,  // CSR pointers
                        const uint32_t V);
