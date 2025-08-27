

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cfloat>

#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>



// 2. Kernel to mark unique edges
__global__ void mark_unique_edges(
    const unsigned int* u_sorted,
    const unsigned int* v_sorted,
    bool* flags,
    size_t N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (idx == 0 ||
        u_sorted[idx] != u_sorted[idx-1] ||
        v_sorted[idx] != v_sorted[idx-1])
    {
        flags[idx] = 1;
        //printf("flag!\n");
    } else {
        flags[idx] = 0;
    }
}

__global__ void interleave_kernel(
    const uint32_t* u_array, 
    const uint32_t* v_array, 
    uint32_t* output, 
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[2*idx] = u_array[idx];
        output[2*idx + 1] = v_array[idx];
    }
}

thrust::device_vector<uint32_t> 
make_pairs_kernel(unsigned int* d_u_raw, unsigned int* d_v_raw, int E) 
{
    thrust::device_vector<uint32_t> edges(2*E);
    
    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    
    interleave_kernel<<<blocks, threads>>>(
        d_u_raw, d_v_raw, 
        thrust::raw_pointer_cast(edges.data()), 
        E
    );
    
    return edges;
}

thrust::device_vector<uint32_t> extract_edges_from_pairs(std::vector<uint32_t>& e0, std::vector<uint32_t>& e1){

    // -- init/allocate --
    int N = e0.size();
    //printf("N: %d\n",N);
    if (N == 0){
        printf("Error -- no edges at all.\n");
        exit(1);
    }
    thrust::device_vector<unsigned int> d_u = e0;
    thrust::device_vector<unsigned int> d_v = e1;
    thrust::device_vector<unsigned int> d_u_sorted(N,0);
    thrust::device_vector<unsigned int> d_v_sorted(N,0);
    thrust::device_vector<bool> d_flags(N,0);
    thrust::device_vector<unsigned int> d_u_tmp(N,0);
    thrust::device_vector<unsigned int> d_v_tmp(N,0);


    //
    //  -- Part 1: Radix Sort By Pair --
    //

    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Determine temp storage size
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_v.data()),
        thrust::raw_pointer_cast(d_v_tmp.data()),
        thrust::raw_pointer_cast(d_u.data()),
        thrust::raw_pointer_cast(d_u_tmp.data()),
        N
    );
    cudaMalloc(&d_temp, temp_bytes);

    // Perform sorting
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_v.data()),
        thrust::raw_pointer_cast(d_v_tmp.data()),
        thrust::raw_pointer_cast(d_u.data()),
        thrust::raw_pointer_cast(d_u_tmp.data()),
        N
    );
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_u_tmp.data()),
        thrust::raw_pointer_cast(d_u_sorted.data()),
        thrust::raw_pointer_cast(d_v_tmp.data()),
        thrust::raw_pointer_cast(d_v_sorted.data()),
        N
    );
    cudaFree(d_temp);

    //
    // -- Part 2: Deduplicate Since Pairs are Inorder --
    //

    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    mark_unique_edges<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_u_sorted.data()),
        thrust::raw_pointer_cast(d_v_sorted.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        N
    );
    //cudaDeviceSynchronize();

    // -- view --
    // {
    // thrust::host_vector<unsigned int> d_u_view = d_u_sorted;
    // thrust::host_vector<unsigned int> d_v_view = d_v_sorted;
    // thrust::host_vector<bool> d_flags_view = d_flags;    
    // for (int i = 0; i < 20; i++){
    //     printf("[%d] (%d,%d,%d)\n",i,d_u_view[i],d_v_view[i],d_flags_view[i]);
    // }
    // for (int i = N-10; i < N; i++){
    //     printf("[%d] (%d,%d,%d)\n",i,d_u_view[i],d_v_view[i],d_flags_view[i]);
    // }
    // }


    // 4. Use CUB to compact unique edges
    d_temp = nullptr; // reset
    temp_bytes = 0; // reset
    unsigned int* d_num_selected;
    cudaMalloc(&d_num_selected, sizeof(unsigned int));
    cudaMemset(d_num_selected, 0, sizeof(unsigned int));

    // Step 4a: determine temp storage size
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_u_sorted.data()),
        //thrust::raw_pointer_cast(d_u_unique.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        d_num_selected,
        N
    );
    cudaMalloc(&d_temp, temp_bytes);
    
    // Step 4b: compact
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_u_sorted.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        d_num_selected,
        N
    );
    unsigned int nedges;
    cudaMemcpy(&nedges,d_num_selected,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(d_v_sorted.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        d_num_selected,
        N
    );

    //unsigned int nedges;
    //cudaMemcpy(&nedges,d_num_selected,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaFree(d_num_selected);
    cudaFree(d_temp);


    // -- view --
    // {
    // thrust::host_vector<unsigned int> d_u_view = d_u_sorted;
    // thrust::host_vector<unsigned int> d_v_view = d_v_sorted;
    // for (int i = 0; i < 20; i++){
    //     printf("[%d] (%d,%d)\n",i,d_u_view[i],d_v_view[i]);
    // }
    // for (int i = nedges-10; i < nedges; i++){
    //     printf("[%d] (%d,%d)\n",i,d_u_view[i],d_v_view[i]);
    // }
    // }
    //cudaDeviceSynchronize();

    // -- Part 3: Read # Unique Edges and Reformat Output --
    unsigned int* d_u_ptr = thrust::raw_pointer_cast(d_u_sorted.data());
    unsigned int* d_v_ptr = thrust::raw_pointer_cast(d_v_sorted.data());
    thrust::device_vector<uint32_t> edges = make_pairs_kernel(d_u_ptr,d_v_ptr,nedges);
    //printf("nedges: %d\n",nedges);
    if (nedges ==0){
        printf("No edges!\n");
        exit(1);
    }

    // -- view --
    // thrust::host_vector<uint32_t> edges_cpu = edges;
    // for (int ix = 0; ix < 10; ix++){
    //     printf("edges_cpu[%d] = (%d,%d)\n",ix,edges_cpu[2*ix+0],edges_cpu[2*ix+1]);
    // }
    // for (int ix = nedges-10; ix < nedges; ix++){
    //     printf("edges_cpu[%d] = (%d,%d)\n",ix,edges_cpu[2*ix+0],edges_cpu[2*ix+1]);
    // }

    return edges;
}