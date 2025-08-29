
// -- basic --
#include <assert.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- project imports --
#include "seg_utils_3d.h"
#include "init_utils.h"


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512



/**********************************************

            Find Border Pixels

**********************************************/


__host__ void set_border(const uint32_t* spix, bool* border,
                        const uint32_t* csr_edges, // 1-hop neighbor data
                        const uint32_t* csr_ptr,  // CSR pointers
                        const uint32_t V){
    
  int num_block = ceil( double(V) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK);
  dim3 BlockPerGrid(num_block);
  cudaMemset(border, 0, V*sizeof(bool));
  find_border_vertex<<<BlockPerGrid,ThreadPerBlock>>>(spix,border,csr_edges,csr_ptr,V);
}

__global__  void find_border_vertex(const uint32_t* spix, bool* border,
                                    const uint32_t* csr_edges, // 1-hop neighbor data
                                    const uint32_t* csr_ptr,  // CSR pointers
                                    const uint32_t V){

    // -- raster vertex --
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  
    if (vertex>=V) return; 
    uint32_t start = csr_ptr[vertex];
    uint32_t end = csr_ptr[vertex+1];
    uint32_t my_spix = spix[vertex];

    bool is_border = false;

    for (uint32_t index=start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        uint32_t neigh_spix = spix[neigh];
        if (neigh_spix == my_spix){ continue; }
        is_border = true;
        break;
    }

    border[vertex] = is_border;

    return;        
}


/**********************************************

         Find Border Pixels @ END

**********************************************/


__host__ void set_border_end(const uint32_t* spix, bool* border,
                            const uint32_t* csr_edges, // 1-hop neighbor data
                            const uint32_t* csr_ptr,  // CSR pointers
                            const uint32_t V){
  int num_block = ceil( double(V) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK);
  dim3 BlockPerGrid(num_block);
  cudaMemset(border, 0, V*sizeof(bool));
  find_border_vertex_end<<<BlockPerGrid,ThreadPerBlock>>>(spix,border,csr_edges,csr_ptr,V);
}

__global__  void find_border_vertex_end(const uint32_t* spix, bool* border,
                                        const uint32_t* csr_edges, // 1-hop neighbor data
                                        const uint32_t* csr_ptr,  // CSR pointers
                                        const uint32_t V){

    // -- raster vertex --
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  
    if (vertex>=V) return; 
    uint32_t start = csr_ptr[vertex];
    uint32_t end = csr_ptr[vertex+1];
    uint32_t my_spix = spix[vertex];

    bool is_border = false;

    for (uint32_t index=start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        uint32_t neigh_spix = spix[neigh];
        if (neigh_spix == my_spix){ continue; }
        else if( neigh_spix > my_spix){ continue; }
        is_border = true;
        break;
    }
    

    border[vertex] = is_border;

    return;        
}
