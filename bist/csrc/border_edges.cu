


/******************************************************************
 * 
 *  Extract only the edges along the border so they can be 
 *  colored and thickened for neat visualizations.
 * 
 ********************************************************************/


#include "border_edges.h"
#include <cub/cub.cuh>


#define THREADS_PER_BLOCK 512


__global__ void
mark_edges(bool* marks, bool* marks_v2, const uint32_t*spix, const bool* border, const uint32_t* edges, const uint32_t E){
	uint32_t edge_index = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (edge_index>=E) return;
    uint32_t vertex0 = edges[2*edge_index];
    uint32_t vertex1 = edges[2*edge_index+1];
    bool both_edges = border[vertex0] && border[vertex1] && (spix[vertex0] == spix[vertex1]);
    marks[2*edge_index+0] = both_edges;
    marks[2*edge_index+1] = both_edges;
    marks_v2[edge_index] = both_edges;
}


__global__ void
edge_counts(int* counts, const uint32_t* edges, const uint8_t* edge_batch_ids, const uint32_t E){
	uint32_t edge_index = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (edge_index>=E) return;
    uint32_t vertex0 = edges[2*edge_index];
    uint8_t batch_index = edge_batch_ids[edge_index];
    //uint32_t vertex1 = edges[2*edge_index+1];
    atomicAdd(&counts[batch_index],1);
}


std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<int>,thrust::device_vector<uint8_t>>
get_border_edges(uint32_t* spix, bool* border, uint32_t* edges, uint8_t* edge_batch_ids, uint32_t B, uint32_t E){
    
    //
    // Part 1: Mark Edges to Keep
    //

    // -- launch parameters --
    int NumThreads = THREADS_PER_BLOCK;
    int edge_nblocks = ceil( double(E) / double(NumThreads) ); 
    dim3 EdgeBlocks(edge_nblocks);

    // -- mark edges to keep --
    thrust::device_vector<bool> marked(2*E, 0);
    thrust::device_vector<bool> marked_v2(E, 0);
    bool* marked_ptr = thrust::raw_pointer_cast(marked.data());
    bool* marked_v2_ptr = thrust::raw_pointer_cast(marked_v2.data());
    mark_edges<<<EdgeBlocks,NumThreads>>>(marked_ptr,marked_v2_ptr,spix,border,edges,E);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //
    // Part 2: Shift Edges Down
    // 

    // -- create copy of edges --
    thrust::device_vector<uint32_t> border_edges(edges, edges + 2*E);

    // -- allocate memory --
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    unsigned int* d_num_selected;
    cudaMalloc(&d_num_selected, sizeof(unsigned int));
    cudaMemset(d_num_selected, 0, sizeof(unsigned int));

    // -- determine temp storage size --
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(border_edges.data()),
        marked_ptr,d_num_selected,2*E
    );
    cudaMalloc(&d_temp, temp_bytes);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- run --
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(border_edges.data()),
        marked_ptr,d_num_selected,2*E
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- read num to keep --
    unsigned int nedges_twice;
    cudaMemcpy(&nedges_twice,d_num_selected,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaFree(d_num_selected);
    cudaFree(d_temp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- resize to filtered data -- 
    unsigned int nedges = nedges_twice/2;
    border_edges.resize(2*nedges);

    //printf("[border only shrinks it!] E,nedges_twice: %d %d\n",E,nedges_twice);

    
    //
    // Part 3: Shift Edge Batch IDS
    // 

    // -- create copy of edges --
    thrust::device_vector<uint8_t> border_bids(edge_batch_ids, edge_batch_ids + E);

    // -- allocate memory --
    void* d_temp_v2 = nullptr;
    size_t temp_bytes_v2 = 0;
    unsigned int* d_num_selected_v2;
    cudaMalloc(&d_num_selected_v2, sizeof(unsigned int));
    cudaMemset(d_num_selected_v2, 0, sizeof(unsigned int));

    // -- determine temp storage size --
    cub::DeviceSelect::Flagged(
        d_temp_v2, temp_bytes_v2,
        thrust::raw_pointer_cast(border_bids.data()),
        marked_v2_ptr,d_num_selected_v2,E
    );
    cudaMalloc(&d_temp_v2, temp_bytes_v2);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- run --
    cub::DeviceSelect::Flagged(
        d_temp_v2, temp_bytes_v2,
        thrust::raw_pointer_cast(border_bids.data()),
        marked_v2_ptr,d_num_selected_v2,E
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- read num to keep --
    unsigned int nedges_bids;
    cudaMemcpy(&nedges_bids,d_num_selected_v2,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaFree(d_num_selected_v2);
    cudaFree(d_temp_v2);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- resize to filtered data -- 
    //printf("nedges, nedges_bids: %d %d\n",nedges,nedges_bids);
    border_bids.resize(nedges_bids);

    //printf("[border only shrinks it!] E,nedges_twice: %d %d\n",E,nedges_twice);


    //
    // -- Part 4: Read # Unique Edges and Reformat Output --
    //
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- get counts --
    thrust::device_vector<int> counts(B,0);
    int BorderEdgeBlocks = ceil( double(nedges) / double(NumThreads) ); 
    edge_counts<<<BorderEdgeBlocks,NumThreads>>>(thrust::raw_pointer_cast(counts.data()),
                                                 thrust::raw_pointer_cast(border_edges.data()),
                                                 thrust::raw_pointer_cast(border_bids.data()),
                                                 nedges);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // -- get pointer --
    thrust::device_vector<int> ptr(nedges+1,0);
    thrust::inclusive_scan(counts.begin(), counts.end(), ptr.begin() + 1);
    // thrust::device_vector<uint32_t> ptr; //  should be batchsize+1

    // -- view --
    // thrust::host_vector<uint32_t> bedges = border_edges;
    // for (int ix=0; ix < 10; ix++){
    //     printf("bedges[%d] = %d\n",ix,bedges[ix]);
    // }

    return {border_edges,ptr,border_bids};
}


