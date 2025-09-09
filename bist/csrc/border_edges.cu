


/******************************************************************
 * 
 *  Extract only the edges along the border so they can be 
 *  colored and thickened for neat visualizations.
 * 
 ********************************************************************/


#include "border_edges.h"
#include <cub/cub.cuh>

#include "seg_utils_3d.h"


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




void filter_to_border_edges(PointCloudData& data){

    // -- update border --
    data.border.resize(data.V,0);
    cudaMemset(data.border_ptr(), 0, data.V*sizeof(bool));
    set_border(data.labels_ptr(), data.border_ptr(), data.csr_edges_ptr(),data.csr_eptr_ptr(),data.V);

    // -- check --
    int nedges = thrust::count(data.border.begin(), data.border.end(), true);
    if (nedges == 0){
        printf("No edges detected. Are you actually running the clustering algorithm?\n");
        return;
    }

    // -- keep only edges along the border for a "B-" viz --
    thrust::device_vector<uint32_t> border_edges;
    thrust::device_vector<int> border_eptr;
    thrust::device_vector<uint8_t> border_bids;
    // std::tie(border_edges,border_eptr,border_bids) = get_border_edges(params.spix_ptr(),border,edges_dptr,ebids_dptr,data.B,data.E);
    std::tie(border_edges,border_eptr,border_bids) = get_border_edges(data.labels_ptr(),data.border_ptr(),data.edges_ptr(),data.edge_batch_ids_ptr(),data.B,data.E);
    int num_edges = border_edges.size()/2;
    printf("num edges: %d\n",num_edges);
    thrust::host_vector<uint32_t> edges_cpu = border_edges;
    data.edges = std::move(border_edges);
    data.eptr = std::move(border_eptr);
    data.edge_batch_ids = std::move(border_bids);
    return;
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




























__global__ void
get_faces_per_vertex(uint32_t* label_votes, uint32_t* face_counts, uint32_t* dual_labels, uint32_t* faces, uint32_t* csr_eptr, uint32_t F){

    // -- get face index --
    uint32_t face_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (face_index>=F) return;

    // -- face label --
    uint32_t spix = dual_labels[face_index];

    // -- unpack triangle --
    uint32_t v0 = faces[3*face_index+0];
    uint32_t v1 = faces[3*face_index+1];
    uint32_t v2 = faces[3*face_index+2];
    // -- the number of faces that include the particular vertex --
    uint32_t i0 = atomicAdd(&face_counts[v0],1);
    uint32_t i1 = atomicAdd(&face_counts[v1],1);
    uint32_t i2 = atomicAdd(&face_counts[v2],1);

    // -- gather all the faces for each vertex --
    uint32_t s0 = csr_eptr[v0];
    uint32_t s1 = csr_eptr[v1];
    uint32_t s2 = csr_eptr[v2];
    label_votes[s0+i0] = spix;
    label_votes[s1+i1] = spix;
    label_votes[s2+i2] = spix;
}

__global__ void
set_label_by_max_vote(uint32_t* labels, uint32_t* global_vote_values, uint32_t* csr_eptr, uint32_t V){
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t start = csr_eptr[vertex];
    uint32_t end = csr_eptr[vertex+1];
    constexpr int MAX_NEIGHS = 32;
    uint32_t votes[MAX_NEIGHS] = {UINT32_MAX};
    uint32_t values[MAX_NEIGHS] = {UINT32_MAX};
    int num = 0;

    // -- read neighbor votes --
    for(int idx = start; idx < end; idx++){
        values[num] = global_vote_values[idx];
        num++;
    }

    // -- find most frequent value --
    uint32_t best_val = 0;
    int max_count = 0;
    
    for(int i = 0; i < num; i++){
        uint32_t val = values[i];
        int count = 0;

        // count occurrences
        for(int j = 0; j < num; j++){
            if(values[j] == val) count++;
        }

        if(count > max_count){
            max_count = count;
            best_val = val;
        }
    }

    // assign the most frequent value as the label
    labels[vertex] = best_val;
}
// __global__ void
// assign_labels_dual2primal(uint32_t* primal_labels, uint32_t* primal_faces, const uint32_t* dual_labels, uint32_t* csr_faces_edges, uint32_t* csr_face_eptr, uint32_t F){
//     uint32_t primal_face = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (primal_face>=F) return;
//     csr_face_eptr

// }


thrust::device_vector<uint32_t> get_primal_labels(PointCloudData& primal, PointCloudData& dual){

    // -- assign dual assignments to primal vertices via [dual-face <-> primal-vertex] --
    // PointCloudData border = primal.copy();
    //border.labels.resize(border.V,UINT32_MAX);
    thrust::device_vector<uint32_t> labels(primal.V,UINT32_MAX);
    uint32_t* labels_dptr = thrust::raw_pointer_cast(labels.data());
    
    // -- launch parameters --
    thrust::device_vector<uint32_t> votes(2*primal.E,UINT32_MAX);
    thrust::device_vector<uint32_t> counts(primal.V,0);
    uint32_t* votes_dptr = thrust::raw_pointer_cast(votes.data()); 
    uint32_t* counts_dptr = thrust::raw_pointer_cast(counts.data()); 
    int NumThreads = THREADS_PER_BLOCK;
    int PrimalFaceBlock = ceil( double(primal.F) / double(NumThreads) ); 
    get_faces_per_vertex<<<PrimalFaceBlock,NumThreads>>>(votes_dptr,counts_dptr,dual.labels_ptr(),primal.faces_ptr(),primal.csr_eptr_ptr(),primal.F);

    // -- set to max vote --
    int PrimalVertexBlock = ceil( double(primal.V) / double(NumThreads) ); 
    set_label_by_max_vote<<<PrimalVertexBlock,NumThreads>>>(labels_dptr,votes_dptr,primal.csr_eptr_ptr(),primal.V);

    return labels;
}



// __global__ void
// mark_exterior_vertex_and_edges(bool* vmarks, bool* emarks, uint32_t* nedges, uint32_t* labels, uint32_t* edges, uint32_t* eptr, uint32_t* dual_vnames, uint32_t V, uint32_t dualF){

//     // we are really marking which faces in the dual graph are on the exterior
//     // vmarks[dual_face] is marking... which faces in the dual graph are on the exterior
//     // emarks[...]       is marking... which edges connected exterior faces in the dual graph
//     // nedges[dual_face] is marking... how many neighbors each face in the dual graph has.. pretty unncessary...
//     // 
//     // dual_vnames simple translates the compacted "face" index into the original "vertex" index 

//     uint32_t dual_face = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (dual_face>=dualF) return;
//     uint32_t vertex = dual_vnames[dual_face];
//     uint32_t start = eptr[vertex];
//     uint32_t end = eptr[vertex+1];

//     uint32_t my_label = labels[vertex];

//     int num_neq = 0;
//     bool matches;
//     bool all_same = true;
//     for(int index = start; index < end; index++){
//         uint32_t neigh = edges[index];
//         matches = labels[neigh] == my_label;
//         all_same = all_same && matches;
//         if (!matches){
//             //emarks[index] = true;
//             //num_neq++;
//         }
//     }

//     //nedges[dual_face] = num_neq;
//     vmarks[dual_face] = !all_same;
// }


// __global__ void
// mark_exterior_step2(uint32_t* faces, uint32_t* face_eptr, bool* vmarks, bool* emarks, uint32_t* nedges, uint32_t* dual_edges, uint32_t* dual_eptr, uint32_t* vnames, uint32_t dual_V, uint32_t dual_F){

//     // we are really marking which faces in the dual graph are on the exterior
//     // vmarks[dual_face] is marking... which faces in the dual graph are on the exterior
//     // emarks[...]       is marking... which edges connected exterior vertex in the dual graph
//     // nedges[dual_face] is marking... how many neighbors each face in the dual graph has.. pretty unncessary...
//     // 
//     // dual_vnames simple translates the compacted "face" index into the original "vertex" index 

//     uint32_t face = threadIdx.x + blockIdx.x * blockDim.x;  // the label
// 	if (face>=dual_F) return;
//     // uint32_t start = face_eptr[face];
//     // uint32_t end   = face_eptr[face+1];
//     // for (int index = start; index < end; index++){
//     // }
//     uint32_t vertex = dual_vnames[dual_face];


//     uint32_t start = dual_eptr[vertex];
//     uint32_t end = dual_edges[vertex+1];
//     if (!vmarks[vertex]){ return; }

//     int num_neq = 0;
//     bool matches;
//     bool all_same = true;
//     for(int index = start; index < end; index++){
//         uint32_t neigh = edges[index];
//         matches = labels[neigh] == my_label;
//         all_same = all_same && matches;
//         if (!matches){
//             emarks[index] = true;
//             num_neq++;
//         }
//     }

//     nedges[dual_face] = num_neq;
//     vmarks[dual_face] = !all_same;
// }

// __global__ void bincount_kernel(uint32_t* counts, uint32_t* labels, uint32_t V, uint32_t S) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= V) return;
//     uint32_t val = labels[idx];
//     if (val < UINT32_MAX){
//         atomicAdd(&counts[val], 1);
//     }
// }

// __global__ void fill_faces(uint32_t* faces, uint32_t* sptr, uint32_t* offsets, uint32_t* labels, bool* vmask, uint32_t S, uint32_t V){
//     int vertex = blockIdx.x * blockDim.x + threadIdx.x;
//     if (vertex >= V) return;
//     if (!vmask[vertex]){ return; }

//     uint32_t spix = labels[vertex];
//     if (spix >= S){ return; } 
//     uint32_t offset = atomicAdd(&offsets[spix],1);
//     uint32_t start = sptr[spix];
//     uint32_t end = sptr[spix+1];
//     uint32_t write_index = start + offset;
//     // if (write_index >= end){
//     //     printf("v,spix,offset: %d, %d, %d, %d, %d\n",vertex,spix,offset,start,end);
//     // }
//     assert(write_index < end);
//     faces[write_index] = vertex;

// }

// __global__ void polygonize_faces(uint32_t* faces, uint32_t* sptr, bool* details_flag, bool* spix_flag, 
//                                 bool* emarks, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t* S, uint32_t Smax){

//     // -- get superpixel vertex --
//     int spix = blockIdx.x * blockDim.x + threadIdx.x;
//     if (spix >= Smax) return;

//     // -- allocate --
//     constexpr int MAX_NEIGHS = 1024;
//     uint32_t neighs[MAX_NEIGHS] = {UINT32_MAX};
//     bool local_emarks[MAX_NEIGHS] = {false};

//     // -- get endpoints --
//     uint32_t start = sptr[spix];
//     uint32_t end = sptr[spix+1];
//     int poly_size = end - start;
//     if (poly_size == 0){ return ;}
//     // if (poly_size >= (MAX_NEIGHS-1)){
//     //     printf("poly_size: %d\n",poly_size);
//     // }
//     assert(poly_size < (MAX_NEIGHS-1));

//     // -- read to memory --
//     int num = 0;
//     for (int index = start; index < end; index++){
//         neighs[num] = faces[index];
//         if (neighs[num] == UINT32_MAX){ break; }
//         num++;
//     }
//     neighs[num] = neighs[0];

//     // -- path traversal --
//     bool success;
//     int selected_ix;
//     for(int ix = 0; ix < num; ix++){
        
//         success = false;
//         selected_ix = -1;
//         uint32_t face = neighs[ix];
//         uint32_t prev = (ix > 0) ? neighs[ix-1] : UINT32_MAX;
//         uint32_t start = csr_eptr[face];
//         uint32_t end = csr_eptr[face+1];

//         // -- read all the bools: "Is this edge on the border?" --
//         assert( (end - start)  < MAX_NEIGHS);
//         assert( (end - start)  <= 3);
//         int _emark_ix = 0;
//         for (uint32_t index = start; index < end; index++){
//             local_emarks[_emark_ix] = emarks[index];
//             _emark_ix++;
//         }
//         //printf("start,end %d %d || le[0], le[1], le[2]: %d %d %d\n",start,end,local_emarks[0],local_emarks[1],local_emarks[2]);
        

//         // -- Find a Neighbors of Vertex "face" that is Part of the Shuffled Polygon --
//         _emark_ix = 0;
//         for (uint32_t index = start; index < end; index++){

//             bool border_edge = local_emarks[_emark_ix];
//             _emark_ix++;
//             //if (!border_edge){ continue; }

//             uint32_t neigh = csr_edges[index];
//             assert(neigh < UINT32_MAX);
//             for (int jx = ix+1; jx <= num; jx++){
//                 if ((neigh == neighs[jx]) && (neigh != prev)){
//                     success = true;
//                     selected_ix = jx;
//                     break;
//                 }
//             }
//             if (success){ break; }
//         }

//         // -- put in order --
//         if (!success){ 
//             atomicAdd(S,1);
//             return; 
//         }
//         int tmp = neighs[selected_ix];
//         neighs[selected_ix] = neighs[ix+1];
//         neighs[ix+1] = tmp; 
        
//     }
    
//     // -- if bool is not set, it failed to close the loop --
//     if (!success){ 
//         return; 
//     }
//     return;
//     atomicAdd(S,1);
    
//     // -- mark for keeping and write if successful --
//     num = 0;
//     start = sptr[spix];
//     end = sptr[spix+1];
//     //end = start + num;
//     for (int index = start; index < end; index++){
//         if (neighs[num+1] == UINT32_MAX){ break; }
//         faces[index] = neighs[num];
//         details_flag[index] = true;
//         num++;
//     }
//     spix_flag[spix] = true;

// }

__global__ void
count_edges_by_spix(uint32_t* spix_edge_count, uint32_t* labels, uint32_t* faces, uint32_t* faces_eptr, uint32_t* vnames ,uint32_t F){
    uint32_t dual_face = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (dual_face>=F) return;
    uint32_t vertex = vnames[dual_face];
    uint32_t label = labels[vertex];
    uint32_t start = faces_eptr[dual_face];
    uint32_t end = faces_eptr[dual_face+1];
    atomicAdd(&spix_edge_count[label],end-start);
}

__global__ void
fill_edges_by_spix(uint32_t* spix_edges, uint32_t* spix_eptr, uint32_t* offsets, uint32_t* labels, uint32_t* faces, uint32_t* faces_eptr, uint32_t* vnames ,uint32_t F){
    uint32_t dual_face = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (dual_face>=F) return;
    uint32_t vertex = vnames[dual_face];
    uint32_t label = labels[vertex];
    uint32_t start = faces_eptr[dual_face];
    uint32_t end = faces_eptr[dual_face+1];
    uint32_t write_start = spix_eptr[label];
    uint32_t offset = atomicAdd(&offsets[label],end-start);
    uint32_t write_index = write_start + offset;
    assert(end - start >= 2);
    uint32_t v0 = faces[start];
    for(int index = start+1; index < end; index++){
        uint32_t v1 = faces[index];
        spix_edges[2*write_index+0] = v0;//min(v0,v1);
        spix_edges[2*write_index+1] = v1;//max(v0,v1);
        // if(label == 1000){
        //     printf("fill[%d] %d (%d, %d)\n",label,dual_face,v0,v1);
        // }        
        v0 = v1;
        write_index++;

    }
    spix_edges[2*write_index+0] = faces[end-1];//min(v0,v1);
    spix_edges[2*write_index+1] = faces[start];//max(v0,v1);
    // if(label == 1000){
    //     printf("fill[%d] %d (%d, %d)\n",label,dual_face,faces[end-1],faces[start]);
    // }  
    //printf("start, end: %d %d\n",faces[start],faces[end]);
}


// -- a little silly; probably a better way --
__global__ void
mark_repeated_edges(uint32_t* nedges_by_spix, bool* edge_marks, uint32_t* spix_edges, uint32_t* spix_eptr, 
                    bool* csr_mask, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t S){

    // -- get superpixel vertex --
    int spix = blockIdx.x * blockDim.x + threadIdx.x;
    if (spix >= S) return;

    uint32_t start = spix_eptr[spix];
    uint32_t end = spix_eptr[spix+1];

    for(int index = start; index < end; index++){

        uint32_t v0_i = spix_edges[2*index+0];
        uint32_t v1_i = spix_edges[2*index+1];
        bool is_this_uniq = true;

        for(int jndex = start; jndex < end; jndex++){
            if( index == jndex ){ continue; }
            uint32_t v0_j = spix_edges[2*jndex+0];
            uint32_t v1_j = spix_edges[2*jndex+1];

            bool cond_a = (v0_i == v0_j) && (v1_i == v1_j);
            bool cond_b = (v1_i == v0_j) && (v0_i == v1_j);
            if( cond_a || cond_b ){
                is_this_uniq = false;
                // edge_marks[index] = false;
                // edge_marks[jndex] = false;
            }
        }

        // -- info --
        // if(spix == 1000){
        //     printf("[%d|%d] (%d, %d)\n",spix,is_this_uniq,v0_i,v1_i);
        // }

        edge_marks[index] = is_this_uniq;

        // -- mark on the csr edges --
        uint32_t csr_start = csr_eptr[v0_i];
        uint32_t csr_end = csr_eptr[v0_i+1];
        for(int csr_index = csr_start; csr_index < csr_end; csr_index++){
            uint32_t csr_neigh = csr_edges[csr_index];
            if (csr_neigh == v1_i){
                csr_mask[csr_index] = is_this_uniq;
                // if(spix == 1000){
                //     printf("(%d,%d) csr_mask[%d] = %d\n",v0_i,v1_i,csr_index,is_this_uniq);
                // }
                break;
            }
        }

        // -- mark on the csr edges --
        csr_start = csr_eptr[v1_i];
        csr_end = csr_eptr[v1_i+1];
        for(int csr_index = csr_start; csr_index < csr_end; csr_index++){
            uint32_t csr_neigh = csr_edges[csr_index];
            if (csr_neigh == v0_i){
                csr_mask[csr_index] = is_this_uniq;
                // if(spix == 1000){
                //     printf("(%d,%d) csr_mask[%d] = %d\n",v0_i,v1_i,csr_index,is_this_uniq);
                // }
                break;
            }
        }



    }

    uint32_t num = 0;
    for(int index = start; index < end; index++){
        num = num + (edge_marks[index] == true);
    }
    nedges_by_spix[spix] = num;
}

__global__ void count_csr_edges_mask(uint32_t* counts, bool* csr_mask, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t V){

    // -- get superpixel vertex --
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    uint32_t start = csr_eptr[vertex];
    uint32_t end = csr_eptr[vertex+1];
    uint32_t count = 0;

    // -- mark on the csr edges --
    for(int index = start; index < end; index++){
        // uint32_t neigh = csr_edges[index];
        count += csr_mask[index] == true;
    }
    counts[vertex] = count;

}

__global__ void uniq_edge_pairs(uint32_t* edge_pairs, uint32_t* nedge_pairs, bool* csr_mask, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t V){
    
    // -- get superpixel vertex --
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    uint32_t start = csr_eptr[vertex];
    uint32_t end = csr_eptr[vertex+1];

    // -- mark on the csr edges --
    for(int index = start; index < end; index++){
        if (!csr_mask[index]){ continue; }
        uint32_t neigh = csr_edges[index];
        if(vertex >= neigh){ continue; }
        uint32_t write_index = atomicAdd(nedge_pairs,1);
        edge_pairs[2*write_index+0] = vertex;
        edge_pairs[2*write_index+1] = neigh;
    }
}

// __global__ void fill_faces(uint32_t* nverts_by_spix, uint32_t* spix_verts, bool* edge_marks, uint32_t* spix_edges, uint32_t* spix_eptr, uint32_t S){

//     // -- get superpixel vertex --
//     int spix = blockIdx.x * blockDim.x + threadIdx.x;
//     if (spix >= S) return;

//     uint32_t start = spix_eptr[spix];
//     uint32_t end = spix_eptr[spix+1];
//     uint32_t write_index = start;

//     for(int index = start; index < end; index++){

//         if (!edge_marks[index]) { continue; } 
//         uint32_t v0 = spix_edges[2*index+0];
//         uint32_t v1 = spix_edges[2*index+1];
        
//         spix_verts[write_index] = v0;
//         spix_verts[write_index+1] = v1;
//         write_index = write_index + 2;
//     }

//     uint32_t num = 0;
//     for(int index = start; index < end; index++){
//         num = num + (edge_marks[index] == true);
//     }
//     nverts_by_spix[spix] = num;
// }

// -- pretty dumb
__global__ void gather_exterior_vertex(uint32_t* labels, uint32_t* spix_verts, uint32_t* nverts_by_spix, bool* edge_marks, uint32_t* spix_edges, uint32_t* spix_eptr, uint32_t S, uint32_t V){
    // -- get superpixel vertex --
    int spix = blockIdx.x * blockDim.x + threadIdx.x;
    if (spix >= S) return;
    
    uint32_t start = spix_eptr[spix];
    uint32_t end = spix_eptr[spix+1];
    uint32_t unique_count = 0;
    constexpr int MAX_VERTS = 1024;
    uint32_t verts[MAX_VERTS] = {UINT32_MAX};
    uint32_t counts[MAX_VERTS] = {0};
    
    // Process each marked edge and add unique vertices
    for(int index = start; index < end; index++){
        if (!edge_marks[index]) { continue; }
        
        uint32_t v0 = spix_edges[2*index+0];
        uint32_t v1 = spix_edges[2*index+1];
        if (v0 == UINT32_MAX){ continue; }
        
        // Check and add v0 if unique
        bool v0_exists = false;
        for(uint32_t j = 0; j < unique_count; j++){
            if(verts[j] == v0){
                v0_exists = true;
                counts[j] += 1;
                break;
            }
        }
        if(!v0_exists){
            verts[unique_count] = v0;
            counts[unique_count] = 1;
            unique_count++;
        }
        assert(unique_count < MAX_VERTS);
        
        // Check and add v1 if unique
        bool v1_exists = false;
        for(uint32_t j = 0; j < unique_count; j++){
            if(verts[j] == v1){
                v1_exists = true;
                counts[j] += 1;
                break;
            }
        }
        if(!v1_exists){
            verts[unique_count] = v1;
            counts[unique_count] = 1;
            unique_count++;
        }
        assert(unique_count < MAX_VERTS);

    }

    uint32_t nkeep = 0;
    for(int index = 0; index < unique_count; index++){
        if(counts[index]==1){ continue; } // skip hairy edges
        spix_verts[start+nkeep] = verts[index];
        assert(verts[index] < V);
        labels[verts[index]] = spix;
        nkeep++;
    }
    
    nverts_by_spix[spix] = nkeep;
}

__global__ void polygonize_faces(uint32_t* faces, bool* mask_vertex_for_face, bool* mask_is_poly, uint32_t* vptr, 
                                uint32_t* spix_verts, uint32_t* spix_edges, uint32_t* sptr_for_edges, uint32_t* nverts_by_spix, 
                                bool* csr_mask, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t* S, uint32_t Smax){


    // -- get superpixel vertex --
    int spix = blockIdx.x * blockDim.x + threadIdx.x;
    if (spix >= Smax) return;

    // -- allocate --
    constexpr int MAX_NEIGHS = 1024;
    uint32_t neighs[MAX_NEIGHS] = {UINT32_MAX};
    uint32_t neighs2[3] = {UINT32_MAX};

    // -- get endpoints --
    uint32_t start = sptr_for_edges[spix];
    uint32_t end = sptr_for_edges[spix+1];
    uint32_t nverts = nverts_by_spix[spix];
    if (nverts == 0){ return ;}
    assert(nverts < (MAX_NEIGHS-1));

    // -- read to memory --
    int num = 0;
    for (int ix = 0; ix < nverts; ix++){
        neighs[num] = spix_verts[start+num];
        if (neighs[num] == UINT32_MAX){ break; }
        num++;
    }
    neighs[num] = neighs[0];

    // -- info --
    if(spix == 1000){
        printf("num,nverts %d %d\n",num,nverts);
        for (int ix = 0; ix < num; ix++){
            printf("neighs[%d] = %d\n",ix,neighs[ix]);
        }
    }

    // -- path traversal --
    bool success;
    int selected_ix;
    for(int ix = 0; ix < num; ix++){
        
        success = false;
        selected_ix = -1;
        uint32_t vertex = neighs[ix];
        uint32_t prev = (ix > 0) ? neighs[ix-1] : UINT32_MAX;
        uint32_t start = csr_eptr[vertex];
        uint32_t end = csr_eptr[vertex+1];

        // -- read all the bools: "Is this edge on the border?" --
        assert( (end - start)  < MAX_NEIGHS);
        assert( (end - start)  <= 3);
        assert( (end - start) > 1); // check if we have any tails.

        // -- Find a Neighbors of Vertex "face" that is Part of the Shuffled Polygon --
        for (uint32_t index = start; index < end; index++){
            if(spix == 1000){ printf("search (%d -> %d) [%d?]\n",vertex,csr_edges[index],csr_mask[index]); }
            if(!csr_mask[index]){ continue; }
            uint32_t neigh = csr_edges[index];
            assert(neigh < UINT32_MAX);
            for (int jx = ix+1; jx <= num; jx++){
                if ((neigh == neighs[jx]) && (neigh != prev)){
                    success = true;
                    selected_ix = jx;
                    break;
                }
            }
            if (success){ break; }
        }

        // -- put in order --
        if (!success){ 
            // atomicAdd(S,1);
            return; 
        }
        int tmp = neighs[selected_ix];
        neighs[selected_ix] = neighs[ix+1];
        neighs[ix+1] = tmp; 
        
    }
    
    // -- if bool is not set, it failed to close the loop --
    if (!success){ 
        return; 
    }
    // return;
    atomicAdd(S,1);
    
    // -- mark for keeping and write if successful --
    num = 0;
    start = vptr[spix];
    end = vptr[spix+1];
    //end = start + num;
    for (int index = start; index < end; index++){
        if (neighs[num+1] == UINT32_MAX){ break; }
        faces[index] = neighs[num];
        mask_vertex_for_face[index] = true;
        num++;
    }
    mask_is_poly[spix] = true;

}


__global__ void set_face_colors(float3* fcolors, uint32_t* face_labels, float3* spix_colors, uint32_t F, uint32_t S){
    // -- get superpixel vertex --
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= F) return;
    uint32_t label = face_labels[face];
    if (label >= S){ 
        float3 empty = make_float3(0.0f,0.0f,0.0f);
        fcolors[face] = empty;
    }else{
        fcolors[face] = spix_colors[label];
    }
}

__global__ void set_vertex_colors(float3* colors, uint32_t* vertex_labels, float3* spix_colors, uint32_t V, uint32_t S){
    // -- get superpixel vertex --
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    uint32_t label = vertex_labels[vertex];
    if (label >= S){ 
        float3 empty = make_float3(0.0f,0.0f,0.0f);
        colors[vertex] = empty;
    }else{
        colors[vertex] = spix_colors[label];
    }
}

void apply_spix_pooling(PointCloudData& data, SuperpixelParams3d& spix_params){
    thrust::device_vector<float3> vertex_colors(data.V);
    uint32_t S = spix_params.nspix_sum;
    {
        float3* vertex_colors_dptr = thrust::raw_pointer_cast(vertex_colors.data());
        int NumThreads = 256;
        int NumVertexColor = ceil( double(data.V) / double(NumThreads) );
        set_vertex_colors<<<NumVertexColor,NumThreads>>>(vertex_colors_dptr, spix_params.spix_ptr(), spix_params.mu_app_ptr(), data.V, S);
    }
    data.ftrs = vertex_colors;
}

PointCloudData get_border_data(PointCloudData& primal, PointCloudData& dual, SuperpixelParams3d& spix_params, uint32_t S){

    // -- get labels for border --
    auto labels = get_primal_labels(primal,dual);
    uint32_t* labels_dptr = thrust::raw_pointer_cast(labels.data());
    PointCloudData border = primal.copy();
    border.labels = labels;
    primal.labels = labels;
    dual.face_labels = labels;

    thrust::device_vector<float3> face_colors(dual.F);
    {
        float3* face_colors_dptr = thrust::raw_pointer_cast(face_colors.data());
        int NumThreads = 256;
        int NumFaceBlocks = ceil( double(dual.F) / double(NumThreads) );
        set_face_colors<<<NumFaceBlocks,NumThreads>>>(face_colors_dptr, labels_dptr, spix_params.mu_app_ptr(), dual.F, S);
    }
    thrust::device_vector<float3> vertex_colors(dual.V);
    {
        float3* vertex_colors_dptr = thrust::raw_pointer_cast(vertex_colors.data());
        int NumThreads = 256;
        int NumVertexColor = ceil( double(dual.V) / double(NumThreads) );
        set_vertex_colors<<<NumVertexColor,NumThreads>>>(vertex_colors_dptr, spix_params.spix_ptr(), spix_params.mu_app_ptr(), dual.V, S);
    }
    dual.face_colors = face_colors;
    dual.ftrs = vertex_colors;

    // thrust::device_vector<uint32_t> new_labels(dual.V,UINT32_MAX);
    // dual.labels = new_labels;

    // -- keep only the boundary edges --
    filter_to_border_edges(border);
    return border;

    // -- removed edges in dual between two primal vertices with the same label --
    // PointCloudData dual = in_dual.copy();

    // -- launch configs --
    int NumThreads = THREADS_PER_BLOCK;
    int DualVertexBlocks = ceil( double(dual.V) / double(NumThreads) );
    int DualFaceBlocks = ceil( double(dual.F) / double(NumThreads) );
    int PrimalVertexBlocks = ceil( double(primal.V) / double(NumThreads) );

    int SpixNumThreads = 64;
    int SpixBlocks = ceil( double(S) / double(SpixNumThreads) ); 


    //
    //
    //  Step 1: Get a List of Dual Edges for Each Spix
    //
    //

    //printf("here!\n");

    thrust::device_vector<uint32_t> spix_edge_count(S,0);
    uint32_t* spix_edge_count_dptr = thrust::raw_pointer_cast(spix_edge_count.data());
    count_edges_by_spix<<<DualFaceBlocks,NumThreads>>>(spix_edge_count_dptr,labels_dptr,dual.faces_ptr(),dual.faces_eptr_ptr(),dual.face_vnames_ptr(),dual.F);
    thrust::device_vector<uint32_t> sptr(S+1,0); // should really be "eptr"
    uint32_t* sptr_dptr = thrust::raw_pointer_cast(sptr.data());
    thrust::inclusive_scan(spix_edge_count.begin(), spix_edge_count.end(), sptr.begin() + 1);
    uint32_t ecount = sptr[S];
    //printf("E_chk dual.E primal.E %d %d %d\n",E, dual.E, primal.E);
    //exit(1);

    thrust::device_vector<uint32_t> spix_edges(2*ecount,UINT32_MAX); // ? x2 or x4?
    uint32_t* spix_edges_dptr = thrust::raw_pointer_cast(spix_edges.data());
    thrust::fill(spix_edge_count.begin(),spix_edge_count.end(),0);
    fill_edges_by_spix<<<DualFaceBlocks,NumThreads>>>(spix_edges_dptr, sptr_dptr, spix_edge_count_dptr, labels_dptr, dual.faces_ptr(),dual.faces_eptr_ptr(),dual.face_vnames_ptr(),dual.F);


    //
    //
    //  Step 2: Mark Repeated Edges within an Spix and (maybe) Remove Them
    //
    //

    // -- mark repeated edges for each spix; these are interior --
    thrust::device_vector<uint32_t> nedges_by_spix(S,0);
    thrust::device_vector<bool> edge_marks(2*ecount,0);
    thrust::device_vector<bool> csr_edge_mask(2*dual.E,0);
    uint32_t* nedges_by_spix_dptr = thrust::raw_pointer_cast(nedges_by_spix.data());
    bool* edge_marks_dptr = thrust::raw_pointer_cast(edge_marks.data());
    bool* csr_edge_mask_dptr = thrust::raw_pointer_cast(csr_edge_mask.data());
    mark_repeated_edges<<<SpixBlocks,SpixNumThreads>>>(nedges_by_spix_dptr, edge_marks_dptr, spix_edges_dptr, sptr_dptr, 
                                                       csr_edge_mask_dptr, dual.csr_edges_ptr(), dual.csr_eptr_ptr(), S);
    int num_boundary_edges = thrust::count(edge_marks.begin(), edge_marks.end(), true);
    
    // -- count number of neighbors for each vertex --
    thrust::device_vector<uint32_t> edge_counts_by_vertex(dual.V,0);
    uint32_t* edge_counts_by_vertex_dptr = thrust::raw_pointer_cast(edge_counts_by_vertex.data());;
    count_csr_edges_mask<<<DualVertexBlocks,NumThreads>>>(edge_counts_by_vertex_dptr, csr_edge_mask_dptr, dual.csr_edges_ptr(), dual.csr_eptr_ptr(), dual.V);
    
    thrust::device_vector<uint32_t> csr_eptr(dual.V+1,0);
    thrust::inclusive_scan(edge_counts_by_vertex.begin(), edge_counts_by_vertex.end(), csr_eptr.begin() + 1);
    //uint32_t* csr_eptr_dptr = thrust::raw_pointer_cast(csr_eptr.data());
    uint32_t E2 = csr_eptr[dual.V];
    printf("E2: %d\n",E2);
    assert (E2 % 2 == 0);
    uint32_t E = E2/2;

    // -- write uniq pairs --
    thrust::device_vector<uint32_t> edges(2*E,0);
    thrust::device_vector<uint32_t> write_counter(1, 0); 
    uint32_t* edges_dptr = thrust::raw_pointer_cast(edges.data());
    uint32_t* write_counter_dptr = thrust::raw_pointer_cast(write_counter.data());;
    uniq_edge_pairs<<<DualVertexBlocks,NumThreads>>>(edges_dptr, write_counter_dptr, csr_edge_mask_dptr, dual.csr_edges_ptr(), dual.csr_eptr_ptr(), dual.V);


    thrust::device_vector<uint32_t> csr_edges = dual.csr_edges;
    uint32_t* csr_edges_dptr = thrust::raw_pointer_cast(csr_edges.data());
    
    // -- compactify valid edges --
    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        thrust::device_vector<uint32_t> dnum(1, 0); 
        unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            csr_edges_dptr, csr_edge_mask_dptr,
            d_num_selected, 2*dual.E);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            csr_edges_dptr, csr_edge_mask_dptr,
            d_num_selected, 2*dual.E);
        cudaFree(d_temp);
    }
    csr_edges.resize(2*E);

    dual.eptr[dual.B] = E;
    dual.E = E;
    dual.csr_edges = std::move(csr_edges);
    dual.csr_eptr = std::move(csr_eptr);
    dual.edges = std::move(edges);

//     // -- enumerate all vertices on exterior --
//     thrust::device_vector<uint32_t> spix_verts(2*E,UINT32_MAX);
//     uint32_t* spix_verts_dptr = thrust::raw_pointer_cast(spix_verts.data());
//     thrust::device_vector<uint32_t> nverts_by_spix(S,0);
//     uint32_t* nverts_by_spix_dptr = thrust::raw_pointer_cast(nverts_by_spix.data());
//     thrust::fill(dual.labels.begin(),dual.labels.end(),UINT32_MAX);
//     gather_exterior_vertex<<<SpixBlocks,SpixNumThreads>>>(dual.labels_ptr(),spix_verts_dptr,nverts_by_spix_dptr,edge_marks_dptr,spix_edges_dptr,sptr_dptr,S,dual.V);

//     // -- count number of each vertex --
    


//     // -- pointer for compactly writing the exterior vertices --
//     thrust::device_vector<uint32_t> vptr(S+1,0);
//     thrust::inclusive_scan(nverts_by_spix.begin(), nverts_by_spix.end(), vptr.begin() + 1);
//     uint32_t* vptr_dptr = thrust::raw_pointer_cast(vptr.data());
//     uint32_t nverts_for_faces = vptr[S];
//     printf("nverts_for_faces: %d\n",nverts_for_faces);

//     // -- reorder the exterior vertices to make a polygon --
//     thrust::device_vector<uint32_t> Spoly_tr(1,0);
//     uint32_t* Spoly_dptr = thrust::raw_pointer_cast(Spoly_tr.data());
//     thrust::device_vector<uint32_t> faces(nverts_for_faces,UINT32_MAX); // at most these many elements to write the faces
//     uint32_t* faces_dptr = thrust::raw_pointer_cast(faces.data());
//     thrust::device_vector<bool> mask_face_vertices(nverts_for_faces,0);
//     bool* mask_face_vertices_dptr = thrust::raw_pointer_cast(mask_face_vertices.data());
//     thrust::device_vector<bool> mask_is_poly(S,0);
//     bool* mask_is_poly_dptr = thrust::raw_pointer_cast(mask_is_poly.data());
//     polygonize_faces<<<SpixBlocks,SpixNumThreads>>>(faces_dptr, mask_face_vertices_dptr,mask_is_poly_dptr,vptr_dptr, 
//                                                     spix_verts_dptr, spix_edges_dptr, sptr_dptr, nverts_by_spix_dptr, 
//                                                     csr_edge_mask_dptr, dual.csr_edges_ptr(), dual.csr_eptr_ptr(), Spoly_dptr, S);
//     uint32_t Spoly = Spoly_tr[0];
//     printf("Spoly: %d\n",Spoly);
//     if (Spoly == 0){
//         printf("No polygons... check this code...\n");
//         exit(1);
//     }
//     // exit(1);

//     // // -- mark vertices with only same neighbors for removal --
//     // assert(dual.F <= primal.V); // we want this!
//     // thrust::device_vector<bool> mark_vertex(dual.F,0);
//     // bool* vmarks_dptr = thrust::raw_pointer_cast(mark_vertex.data());
//     // thrust::device_vector<bool> mark_edges(2*primal.E,0);
//     // bool* emarks_dptr = thrust::raw_pointer_cast(mark_edges.data());
//     // thrust::device_vector<uint32_t> ecounts(dual.F,0);
//     // uint32_t* ecounts_dptr = thrust::raw_pointer_cast(ecounts.data());
//     // //mark_exterior_vertex_and_edges<<<DualVertexBlocks,NumThreads>>>(vmarks_dptr,emarks_dptr,ecounts_dptr,dual.labels_ptr(),dual.csr_edges_ptr(),dual.csr_eptr_ptr(),dual.V);
//     // // exit(1);
//     // mark_exterior_vertex_and_edges<<<DualFaceBlocks,NumThreads>>>(vmarks_dptr,emarks_dptr,ecounts_dptr,labels_dptr,
//     //                                                                 primal.csr_edges_ptr(),primal.csr_eptr_ptr(),dual.face_vnames_ptr(),primal.V,dual.F);
//     // mark_exterior_step2<<<DualFaceBlocks,NumThreads>>>(vmarks_dptr,emarks_dptr,ecounts_dptr,labels_dptr,
//     //                                                                 primal.csr_edges_ptr(),primal.csr_eptr_ptr(),dual.face_vnames_ptr(),primal.V,dual.F);

//     // // -- count number of primal vertex (or number of dual faces) per spix --
//     // printf("S: %d\n",S);
//     // thrust::device_vector<uint32_t> scounts(S,0);
//     // uint32_t* scounts_dptr = thrust::raw_pointer_cast(scounts.data());
//     // bincount_kernel<<<PrimalVertexBlocks,NumThreads>>>(scounts_dptr,primal.labels_ptr(),primal.V,S);

//     // // -- write the "edge" vertices for each faces --
//     // thrust::device_vector<uint32_t> sptr(S + 1, 0);
//     // uint32_t* sptr_dptr = thrust::raw_pointer_cast(sptr.data());
//     // thrust::inclusive_scan(scounts.begin(), scounts.end(), sptr.begin() + 1);
//     // thrust::device_vector<uint32_t> faces(dual.V,UINT32_MAX); // S unique faces consisting of AT MOST "V" points
//     // uint32_t* faces_dptr = thrust::raw_pointer_cast(faces.data());
//     // thrust::fill(scounts.begin(),scounts.end(),0); // reset
//     // fill_faces<<<DualVertexBlocks,NumThreads>>>(faces_dptr,sptr_dptr,scounts_dptr,dual.labels_ptr(),vmarks_dptr,S,dual.V); // one label per dual-face






//     // exit(1);







// //     // -- reorder the vertices in each face to make a polygon using only the border edges --
// //     thrust::device_vector<uint32_t> Spoly_tr(1,0);
// //     uint32_t* Spoly_dptr = thrust::raw_pointer_cast(Spoly_tr.data());
// //     thrust::device_vector<bool> details_flag(dual.V,0);
// //     bool* details_flag_dptr = thrust::raw_pointer_cast(details_flag.data());
// //     thrust::device_vector<bool> mark_spix(S,0);
// //     bool* smarks_dptr = thrust::raw_pointer_cast(mark_spix.data());
// //     polygonize_faces<<<SpixBlocks,SpixNumThreads>>>(faces_dptr, sptr_dptr, details_flag_dptr, smarks_dptr, emarks_dptr, 
// //                                                     dual.csr_edges_ptr(), dual.csr_eptr_ptr(), Spoly_dptr, S);
// //     uint32_t Spoly = Spoly_tr[0];
// //     printf("Spoly: %d\n",Spoly);
// //     if (Spoly == 0){
// //         printf("No polygons... check this code...\n");
// //         exit(1);
// //     }
// //     exit(1);

//     // -- Compactify Faces IDS --
//     {
//         void* d_temp = nullptr;
//         size_t temp_bytes = 0;
//         thrust::device_vector<uint32_t> dnum(1, 0); 
//         unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
//         cub::DeviceSelect::Flagged(
//             d_temp, temp_bytes,
//             faces_dptr, mask_face_vertices_dptr,
//             d_num_selected, nverts_for_faces);
//         cudaMalloc(&d_temp, temp_bytes);
//         cub::DeviceSelect::Flagged(
//             d_temp, temp_bytes,
//             faces_dptr, mask_face_vertices_dptr,
//             d_num_selected, nverts_for_faces);
//         cudaFree(d_temp);
//     }

//     // -- Compactify Size of Each Face --
//     {
//         void* d_temp = nullptr;
//         size_t temp_bytes = 0;
//         thrust::device_vector<uint32_t> dnum(1, 0); 
//         unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
//         cub::DeviceSelect::Flagged(
//             d_temp, temp_bytes,
//             nverts_by_spix_dptr, mask_is_poly_dptr,
//             d_num_selected, S);
//         cudaMalloc(&d_temp, temp_bytes);
//         cub::DeviceSelect::Flagged(
//             d_temp, temp_bytes,
//             nverts_by_spix_dptr, mask_is_poly_dptr,
//             d_num_selected, S);
//         cudaFree(d_temp);
//         uint32_t _tmp = dnum[0];
//         assert(_tmp == Spoly);
//     }
 
//     // -- get offsets per vertex... kind of weird... _nfaces ~= 230k vs F ~= 40k --
//     nverts_by_spix.resize(Spoly);
//     thrust::device_vector<uint32_t> sptr_comp(Spoly+1, 0);
//     thrust::inclusive_scan(nverts_by_spix.begin(), nverts_by_spix.end(), sptr_comp.begin() + 1);
//     uint32_t _num_verts_for_faces = sptr_comp[Spoly];
//     printf("V, _num_verts_for_faces: %d %d\n",dual.V,_num_verts_for_faces);
//     faces.resize(_num_verts_for_faces);

//    // -- silly for now; mostly a lie --
//     // thrust::device_vector<uint8_t> vertex_batch_ids(data.F,0);
//     // thrust::device_vector<uint8_t> edge_batch_ids.resize(0);
//     // dual.vptr[dual.B] = dual.V;
//     dual.fptr[dual.B] = Spoly;
//     dual.faces = std::move(faces);
//     dual.faces_eptr = std::move(sptr_comp);
//     dual.edges.resize(0);
//     //dual.eptr[dual.B] = 0;
//     printf("dual.faces.size(): %d\n",dual.faces.size());

    
    //
    //
    //  After marking vertices and edges for removal, we rename the vertices to account for the deletion
    //
    //

    // // -- for relabling after compact step --
    // thrust::device_vector<uint32_t> vnames(dual.V);
    // thrust::sequence(vnames.begin(), vnames.end(), 0);
    // uint32_t* vnames_dptr = thrust::raw_pointer_cast(vnames.data());

    // // -- Compactify Faces IDS --
    // uint32_t V;
    // {
    //     void* d_temp = nullptr;
    //     size_t temp_bytes = 0;
    //     thrust::device_vector<uint32_t> dnum(1, 0); 
    //     unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
    //     cub::DeviceSelect::Flagged(
    //         d_temp, temp_bytes,
    //         vnames_dptr, vmarks_dptr,
    //         d_num_selected, dual.V);
    //     cudaMalloc(&d_temp, temp_bytes);
    //     cub::DeviceSelect::Flagged(
    //         d_temp, temp_bytes,
    //         vnames_dptr, vmarks_dptr,
    //         d_num_selected, dual.V);
    //     cudaFree(d_temp);
    //     V = dnum[0]; // number of vertices after removing interior
    // }
    // vnames.resize(V); // just frees extra info & keeps old info



    return border;

}