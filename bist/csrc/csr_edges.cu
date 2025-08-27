

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// -- cuda --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

// -- local --
#include "csr_edges.h"
#include "init_utils.h"

// CUDA kernel to count degrees
__global__ void count_degrees_kernel(uint32_t* edges, uint8_t* ebids, int* eptr, int* vptr, uint32_t* degree, int nedges_total) {

    // -- get iterator --
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nedges_total){ return; }

    // -- get index --
    int bx = ebids[idx]; // magic....
    // int E = eptr[bx+1] - eptr[bx];
    int V = vptr[bx+1] - vptr[bx];
    // int edge_batch_offset = 2*eptr[bx];
    int vertex_batch_offset = vptr[bx];

    uint32_t u = edges[idx * 2];
    uint32_t v = edges[idx * 2 + 1];
    if (u < V) atomicAdd(&degree[u+ vertex_batch_offset], 1);
    if (v < V) atomicAdd(&degree[v+ vertex_batch_offset], 1);
    
}


// CUDA kernel to fill CSR edges
__global__ void fill_csr_edges_kernel(uint32_t* edges, uint8_t* ebids,
                                      int* eptr, int* vptr, 
                                      uint32_t* csr_edges, uint32_t* csr_eptr, 
                                      uint32_t* current_pos, int nedges_total) {
    // -- get iterator --
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nedges_total){ return; }
                                    
    // -- get index --
    int bx = ebids[idx]; // magic....
    // int E = eptr[bx+1] - eptr[bx];
    // int V = vptr[bx+1] - vptr[bx];
    //int edge_batch_offset = 2*eptr[bx];
    int vertex_batch_offset = vptr[bx];
    //int vmax = vptr[bx+1];

    // read
    uint32_t u = edges[idx * 2] + vertex_batch_offset;
    uint32_t v = edges[idx * 2 + 1] + vertex_batch_offset; 
    
    // Add edge u -> v
    uint32_t pos_u = atomicAdd(&current_pos[u], 1);
    csr_edges[pos_u] = v;
    
    // Add edge v -> u
    uint32_t pos_v = atomicAdd(&current_pos[v], 1);
    csr_edges[pos_v] = u;
}

std::tuple<uint32_t*, uint32_t*>
get_csr_graph_from_edges(uint32_t* edges, uint8_t* ebids, int* eptr, int* vptr, int V, int E){

    //  -- view --
    //printf("V,E: %d,%d\n",V,E);

    // -- allocate edge pointer --
    uint32_t* csr_eptr = (uint32_t*)easy_allocate((V+1),sizeof(uint32_t));
    cudaMemset(csr_eptr, 0, sizeof(uint32_t)); // first byte is zero

    // -- allocate helpers --
    uint32_t* degree = (uint32_t*)easy_allocate(V,sizeof(uint32_t));
    cudaMemset(degree, 0, V * sizeof(uint32_t));
    uint32_t* curr_pos = (uint32_t*)easy_allocate(V,sizeof(uint32_t));


    //
    //
    // Step 1: Compute the # Degrees & Accumulate
    //
    //

    // Step 1a: Count degrees; edges per vertex
    int block_size = 256;
    int grid_size = (E + block_size - 1) / block_size;
    count_degrees_kernel<<<grid_size, block_size>>>(edges, ebids, eptr, vptr, degree, E);

    // Step 1b:: Compute prefix sum to get csr_eptr
    thrust::inclusive_scan(thrust::device, degree, degree + V, csr_eptr + 1);
    cudaMemcpy(curr_pos, csr_eptr, V * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    //
    //
    // Step 2: Fill CSR edges
    //
    //

    // -- allocate & fill csr edges --
    int num_csr_edges;
    cudaMemcpy(&num_csr_edges,&csr_eptr[V],sizeof(uint32_t),cudaMemcpyDeviceToHost);
    printf("num_csr_edges: %d,%d\n",num_csr_edges,E);
    uint32_t* csr_edges = (uint32_t*)easy_allocate((num_csr_edges+1),sizeof(uint32_t));
    fill_csr_edges_kernel<<<grid_size, block_size>>>(edges, ebids, eptr, vptr, csr_edges, csr_eptr, curr_pos, E);
    
    // -- view --
    // thrust::device_ptr<uint32_t> csr_edges_thp(csr_edges);
    // thrust::device_ptr<uint32_t> csr_eptr_thp(csr_eptr);
    // thrust::host_vector<uint32_t> csr_edges_cpu(csr_edges_thp, csr_edges_thp + num_csr_edges);
    // thrust::host_vector<uint32_t> csr_eptr_cpu(csr_eptr_thp, csr_eptr_thp + V+1);
    // for (int v = 0; v < 5; v++){
    //     int start = csr_eptr_cpu[v];
    //     int end = csr_eptr_cpu[v+1];
    //     printf("[%d]:",v);
    //     for (int ix=start; ix < end; ix++){
    //         printf(" %d,",csr_edges_cpu[ix]);
    //     }
    //     printf("\n");
    // }
    // for (int v = V-5; v < V; v++){
    //     int start = csr_eptr_cpu[v];
    //     int end = csr_eptr_cpu[v+1];
    //     printf("[%d]:",v);
    //     for (int ix=start; ix < end; ix++){
    //         printf(" %d,",csr_edges_cpu[ix]);
    //     }
    //     printf("\n");
    // } 

    // -- cuda free --
    cudaDeviceSynchronize();
    cudaFree(degree);
    cudaFree(curr_pos);

    return std::tuple(csr_edges,csr_eptr);
}


// CUDA kernel to write pairs only if vertex < neigh.
__global__ void write_too_big(uint32_t* edges2big, int* bcount, bool* flag,
                              const uint32_t* csr_edges, const uint32_t* csr_eptr, const uint8_t* vbids, int V) {

    // -- get iterator --
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V){ return; }
    int bx = vbids[vertex];
    int start = csr_eptr[vertex];
    int end = csr_eptr[vertex+1];
    //int nedges = (end - start)/2;

    // -- ... --
    int nedges = 0;
    for(int index=start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        // if (vertex == neigh){
        //     printf("SELF! %d, %d\n",vertex,neigh);
        // }
        if (vertex >= neigh){ continue; }
        edges2big[2*index] = vertex; // - vertex_batch_offset; // one day...
        edges2big[2*index+1] = neigh; // - vertex_batch_offset; // one day...
        flag[2*index] = true;
        flag[2*index+1] = true;
        nedges += 1;
    }
    atomicAdd(&bcount[bx],nedges);

}

std::tuple<uint32_t*, int*>
get_edges_from_csr(uint32_t* csr_edges, uint32_t* csr_eptr, int* vptr, uint8_t* vbids, int V, int B){

    // -- number of total edges --
    int num_csr_edges;
    cudaMemcpy(&num_csr_edges,&csr_eptr[V],sizeof(uint32_t),cudaMemcpyDeviceToHost);
    int E = num_csr_edges/2;

    // -- allocate --
    uint32_t* edges2big = (uint32_t*)easy_allocate(2*num_csr_edges,sizeof(uint32_t));
    bool* flags = (bool*)easy_allocate(2*num_csr_edges,sizeof(bool));
    int* eptr = (int*)easy_allocate((B+1),sizeof(int));
    int* bcount = (int*)easy_allocate(B,sizeof(int));
    cudaMemset(bcount, 0, B*sizeof(int)); // first byte is zero
    cudaMemset(flags, 0, 2*num_csr_edges*sizeof(bool)); // first byte is zero


    // -- Write only (min(e0,e1),max(e0,e1)) pairs; Lots of "empty" space.  --
    int block_size = 256;
    int grid_size = (V + block_size - 1) / block_size;
    write_too_big<<<grid_size, block_size>>>(edges2big, bcount, flags, csr_edges, csr_eptr, vbids, V);
    
    // -- accumulate across eptr --
    cudaMemset(eptr, 0, sizeof(int)); // first byte is zero
    thrust::inclusive_scan(thrust::device, bcount, bcount + B, eptr + 1);

    // -- check --
    int _nedges;
    cudaMemcpy(&_nedges,&eptr[B],sizeof(int),cudaMemcpyDeviceToHost);
    printf("nedges vs _nedges: %d,%d\n",E,_nedges);

    // -- Use CUB to compactify the result. --
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    unsigned int* d_num_selected;
    cudaMalloc(&d_num_selected, sizeof(unsigned int));
    cudaMemset(d_num_selected, 0, sizeof(unsigned int));

    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        edges2big, flags,
        d_num_selected,
        2*num_csr_edges
    );
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        edges2big, flags,
        d_num_selected,
        2*num_csr_edges
    );

    // -- keep only the filled portion --
    uint32_t* edges = (uint32_t*)easy_allocate(2*E,sizeof(uint32_t));
    cudaMemcpy(edges,edges2big,2*E*sizeof(uint32_t),cudaMemcpyDeviceToDevice);

    // -- free --
    cudaFree(bcount);
    cudaFree(d_temp);
    cudaFree(d_num_selected);
    cudaFree(edges2big);
    cudaFree(flags);

    // -- return --
    return std::tuple(edges,eptr);
}


// // Helper function to build CSR from edge list
// void build_csr_from_edges(
//     const uint32_t* edges,      // Input: (E,2) edge pairs
//     uint32_t* csr_edges,        // Output: CSR edge list
//     uint32_t* eptr,             // Output: CSR edge pointers
//     int num_vertices,
//     int num_edges
// ) {
//     // Count degrees
//     std::vector<int> degrees(num_vertices, 0);
//     for (int i = 0; i < num_edges; i++) {
//         degrees[edges[2*i]]++;
//         degrees[edges[2*i + 1]]++;
//     }
    
//     // Build eptr (prefix sum)
//     eptr[0] = 0;
//     for (int v = 0; v < num_vertices; v++) {
//         eptr[v + 1] = eptr[v] + degrees[v];
//     }
    
//     // Fill adjacency list
//     std::vector<int> counters(num_vertices, 0);
//     for (int i = 0; i < num_edges; i++) {
//         uint32_t v1 = edges[2*i];
//         uint32_t v2 = edges[2*i + 1];
        
//         // Add edge v1 -> v2
//         csr_edges[eptr[v1] + counters[v1]++] = v2;
        
//         // Add edge v2 -> v1
//         csr_edges[eptr[v2] + counters[v2]++] = v1;
//     }
// }

// // C-style API
// extern "C" void color_point_cloud_luby(
//     const float3* pos,          // (V,3) point positions (unused in coloring)
//     const uint32_t* edges,      // (E,2) edge pairs
//     uint8_t* group,             // (V,1) output color assignments
//     int num_vertices,
//     int num_edges
// ) {
//     // Build CSR representation
//     std::vector<uint32_t> csr_edges(2 * num_edges);
//     std::vector<uint32_t> eptr(num_vertices + 1);
    
//     build_csr_from_edges(
//         edges,
//         csr_edges.data(),
//         eptr.data(),
//         num_vertices,
//         num_edges
//     );
    
//     // Run Luby's coloring
//     luby_color_graph(
//         csr_edges.data(),
//         eptr.data(),
//         group,
//         num_vertices,
//         csr_edges.size()
//     );
// }

// // Example usage
// int main() {
//     // Example: 8 vertices, 10 edges
//     const int V = 8;
//     const int E = 10;
    
//     // Sample point positions
//     float3 pos[V] = {
//         {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
//         {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
//     };
    
//     // Sample edges
//     uint32_t edges[2*E] = {
//         0,1, 0,2, 0,3,    // vertex 0 connects to 1,2,3
//         1,2, 1,4,         // vertex 1 connects to 2,4
//         2,3, 2,4, 2,5,    // vertex 2 connects to 3,4,5
//         4,5, 5,6          // additional connections
//     };
    
//     uint8_t group[V];
    
//     // Color the graph
//     color_point_cloud_luby(pos, edges, group, V, E);
    
//     // Print results
//     printf("Luby's Coloring Results:\n");
//     for (int i = 0; i < V; i++) {
//         printf("Vertex %d: Color %d\n", i, (int)group[i]);
//     }
    
//     // Count number of colors used
//     int max_color = 0;
//     for (int i = 0; i < V; i++) {
//         if (group[i] > max_color) max_color = group[i];
//     }
//     printf("Number of colors used: %d\n", max_color + 1);
    
//     return 0;
// }
