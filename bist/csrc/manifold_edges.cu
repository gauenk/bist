/*

    Enforce "Manifold" Edges 
    or Fix Non-Manifold Edges

    which are edges connected to more than two faces...

    for this code, we simply split/duplicate?

*/


#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <cub/cub.cuh>

#include "structs_3d.h"
#include "manifold_edges.h"



__global__ void
fill_ve_batch_ids(uint8_t* batch_ids, int* ptr, uint32_t VE, uint32_t B){

    // -- index --
    uint32_t index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= VE){ return; }

    uint8_t batch_index = 0;
    while(batch_index < B){
        if (index >= ptr[batch_index]){ break; }
        batch_index++;
    }
    assert(batch_index<B);
    assert(index >= ptr[batch_index]);
    batch_ids[index] = batch_index;
}



__global__ void
update_batch_info(int* vptr, int* eptr, uint32_t* new_csum, uint32_t B){

    // -- index --
    uint32_t batch_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (batch_index > B){ return; }

    // -- update --
    vptr[batch_index] += new_csum[batch_index];
    eptr[batch_index] += 2*new_csum[batch_index]; // 1 new vertex is 2 new edges
}


__global__ void
copy_edge_info(uint32_t* edges_n, uint8_t* ebids_n, uint32_t* edges, uint8_t* ebids, uint32_t* new_csum, uint32_t E_old){

    // -- index --
    uint32_t edge_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (edge_index >= E_old){ return; }

    // -- ... --
    uint32_t bx = ebids[edge_index];
    uint32_t new_edge_index = edge_index + 2*new_csum[bx]; // every 1 new vertex is 2 new edges

    // -- write --
    edges_n[2*new_edge_index+0] = edges[2*edge_index+0];
    edges_n[2*new_edge_index+1] = edges[2*edge_index+1];
    ebids_n[new_edge_index] = ebids[edge_index];
}


__global__ void
copy_vertex_info(float3* ftrs_n, float3* pos_n, uint8_t* vbids_n, float3* ftrs, float3* pos, uint8_t* vbids, uint32_t* new_csum, uint32_t V_old){

    // -- index --
    uint32_t vertex_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (vertex_index >= V_old){ return; }

    // -- ... --
    uint32_t bx = vbids[vertex_index];
    uint32_t new_vertex = vertex_index + new_csum[bx];

    // -- write --
    ftrs_n[new_vertex] = ftrs[vertex_index];
    pos_n[new_vertex] = pos[vertex_index];
    vbids_n[new_vertex] = vbids[vertex_index];
}

__global__ void
copy_over_edges(uint64_t* uniq_edges, uint32_t* edges, uint32_t E){

    // -- index --
    uint32_t edge_index = threadIdx.x + blockIdx.x*blockDim.x;
    assert(blockDim.x == 512); // should match the "512" below.
    if (edge_index >= E){ return; }  

    // -- read a unique edge pair --
    uint64_t edge_pair = edges[edge_index];
    uint32_t e0 = static_cast<uint32_t>(edge_pair>>32);
    uint32_t e1 = static_cast<uint32_t>(edge_pair);
    edges[2*edge_index+0] = e0;
    edges[2*edge_index+1] = e1;
    
}

__global__ void
count_num_new(uint32_t* num_new, uint8_t* bids, uint64_t* edges, uint32_t* counts, uint32_t E_uniq){
    
    // -- index --
    uint32_t edge_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (edge_index >= E_uniq){ return; }

    // // -- read edges --
    // uint64_t edge_pair = edges[edge_index];
    // uint32_t e0 = static_cast<uint32_t>(edge_pair>>32);
    // uint32_t e1 = static_cast<uint32_t>(edge_pair);

    // -- write amount > 2 --
    uint32_t count = counts[edge_index];
    if (count <= 2){ return; }
    uint8_t bx = bids[edge_index];
    atomicAdd(&num_new[bx],count-2);

}

__global__ void
update_faces_edges_verts(float3* ftrs, float3* pos, uint32_t* edges, uint32_t* faces, uint8_t* fbids, 
                         bool* mask,// uint32_t* fmap,
                         uint8_t* bids, int* vptr, int* eptr, int* fptr, 
                         int* vptr_old, int* eptr_old,
                         uint32_t* v_offset, uint32_t* e_offset, uint32_t F){

    // -- index --
    uint32_t face_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (face_index >= F){ return; }
    int bx = fbids[face_index];

    int vertex_offset = vptr[bx];
    int V_old = vptr_old[bx+1] - vptr_old[bx];
    int vertex_start = vertex_offset + V_old;

    int edge_offset = eptr[bx];
    int E_old = eptr_old[bx+1] - eptr_old[bx];
    int edge_start = edge_offset + E_old;

    // -- write --
    for (int ix = 0; ix < 3; ix++){
        if(!mask[3*face_index+ix]){ continue; }

        // get a new vertex name 
        uint32_t new_vertex = atomicAdd(&v_offset[bx],1)+vertex_start;
        uint32_t old_vertex = faces[3*face_index+ix];

        // -- update face name --
        faces[3*face_index+ix] = new_vertex;

        // -- insert vertex info (position and feature) --
        pos[new_vertex] = pos[old_vertex];
        ftrs[new_vertex] = ftrs[old_vertex];

        // -- insert edge info --
        uint32_t e0 = faces[3*face_index+(ix+1)%3];
        uint32_t e1 = faces[3*face_index+(ix+2)%3];
        uint32_t new_edge = atomicAdd(&e_offset[bx],2)+edge_start;
        //uint32_t edge_index = fmap[3*face_index+ix];
        //uint32_t old_edge = edges[2*new_edge+1];
        // edges[2*edge_index+0] = e1; // definately min
        // edges[2*edge_index+1] = new_vertex; // definately max // maybe "- vertex_offset"? i forget of we offset vert names for uniq edges pairs, but i don't think so...
        edges[2*new_edge+0] = e1; // definately min
        edges[2*new_edge+1] = new_vertex; // definately max // maybe "- vertex_offset"? i forget of we offset vert names for uniq edges pairs, but i don't think so...
        edges[2*new_edge+2] = e0;
        edges[2*new_edge+3] = new_vertex;

        // if(new_vertex == 81399){
        //     printf("[added!] vert[%d -> %d] edge[%d] = (%d,%d) -> (%d,%d) \n",old_vertex,new_vertex,new_edge,old_vertex,e1,e1,new_vertex);
        // }
        //printf("[added!] vert[%d -> %d] edge[%d] = (%d,%d) -> (%d,%d) \n",old_vertex,new_vertex,new_edge,old_vertex,e1,e1,new_vertex);
    }

}

__global__ void
mark_faces_for_update(bool* mask, //uint32_t* fmap, 
                     uint32_t* faces, uint32_t* faces_by_edges, uint32_t* ptr, uint64_t* uniq_edges, 
                      uint8_t* bids, int* vptr, int* fptr, uint32_t E_uniq){

    // -- index --
    uint32_t edge_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (edge_index >= E_uniq){ return; }

    // -- get edge pair --
    uint64_t edge_pair = uniq_edges[edge_index];
    uint32_t e0 = static_cast<uint32_t>(edge_pair>>32);
    uint32_t e1 = static_cast<uint32_t>(edge_pair);
    assert(e0 != e1); // can't be equal.

    // -- endpoints --
    uint32_t bx = bids[edge_index];
    uint32_t vertex_offset = vptr[bx];
    uint32_t face_offset = fptr[bx];
    uint32_t start = ptr[edge_index];
    uint32_t end   = ptr[edge_index+1];

    for (int index=start+2; index < end; index++){
        uint32_t face_index = faces_by_edges[index];
        // uint32_t face_index = faces[index];//+face_offset; // already in encoded faces

        // -- identify problematic edge --
        bool at_least_one = false;
        for (int ix = 0; ix < 3; ix++){
            uint32_t c0_raw = faces[3*face_index+ix] + vertex_offset;
            uint32_t c1_raw = faces[3*face_index+(ix+1)%3] + vertex_offset;
            uint32_t c0 = min(c0_raw,c1_raw);
            uint32_t c1 = max(c0_raw,c1_raw);
            if((e0 == c0) && (e1 == c1)){
                mask[3*face_index+ix] = 1;
                //fmap[3*face_index+ix] = edge_index;
                at_least_one = true;
            }
        }
        // if (at_least_one == true){
        //     uint32_t v0 = faces[3*face_index+0]+vertex_offset;
        //     uint32_t v1 = faces[3*face_index+1]+vertex_offset;
        //     uint32_t v2 = faces[3*face_index+2]+vertex_offset;
        //     printf("(e0,e1) (c0,c1,2): (%d,%d) (%d,%d,%d)\n",e0,e1,v0,v1,v2);
        // }
        assert(at_least_one == true);
    }

}




__global__ void
encode_edges_and_faces(uint64_t* edges, uint64_t* aug_faces, uint32_t* faces, uint8_t* batch_ids, int* vptr, int* fptr, uint32_t F){

    // -- get index --
    uint32_t face_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (face_index >= F){ return; }


    // -- batch info --
    uint8_t batch = batch_ids[face_index];
    uint32_t vertex_offset = vptr[batch];
    uint32_t face_offset = fptr[batch];
    face_index += face_offset; // offset faces for batching

    // -- read --
    uint32_t v0 = faces[3*face_index+0]+vertex_offset; // cast to uint64
    uint32_t v1 = faces[3*face_index+1]+vertex_offset;
    uint32_t v2 = faces[3*face_index+2]+vertex_offset;
    
    // -- standardize order --
    uint32_t e0_min = min(v0,v1);
    uint32_t e0_max = max(v0,v1);
    assert(e0_min != e0_max);
    uint64_t e0_pair = (uint64_t(e0_min) << 32) | uint64_t(e0_max);

    uint32_t e1_min = min(v1,v2);
    uint32_t e1_max = max(v1,v2);
    assert(e1_min != e1_max);
    uint64_t e1_pair = (uint64_t(e1_min) << 32)  | uint64_t(e1_max);


    uint32_t e2_min = min(v2,v0);
    uint32_t e2_max = max(v2,v0);
    assert(e2_min != e2_max);
    uint64_t e2_pair = (uint64_t(e2_min) << 32) | uint64_t(e2_max);

    // -- write --
    edges[3*face_index+0] = e0_pair;
    edges[3*face_index+1] = e1_pair;
    edges[3*face_index+2] = e2_pair;
    aug_faces[3*face_index+0] = face_index;
    aug_faces[3*face_index+1] = face_index;
    aug_faces[3*face_index+2] = face_index;


}

// __global__ void
// count_num_unique(uint64_t* edges, uint32_t* nuniq, uint32_t F3){

//     // -- get face index --
//     uint32_t face_index = threadIdx.x + blockIdx.x*blockDim.x;
//     if (face_index >= F3){ return; }

//     // -- first element is always uniq --
//     if(face_index == 0){
//         atomicAdd(nuniq,1);
//         return;
//     }

//     uint64_t prev = edges[face_index-1];
//     uint64_t curr = edges[face_index];
//     if(prev == curr){ return; }
//     atomicAdd(nuniq,1);

// }

// __global__ void
// write_num_unique(uint64_t* uniq_edges, uint64_t* edges, uint32_t* nuniq, uint32_t F3){

//     // -- get face index --
//     uint32_t face_index = threadIdx.x + blockIdx.x*blockDim.x;
//     if (face_index >= F3){ return; }

//     // -- first element is always uniq --
//     if(face_index == 0){
//         atomicAdd(nuniq,1);
//         return;
//     }

//     uint64_t prev = edges[face_index-1];
//     uint64_t curr = edges[face_index];
//     if(prev == curr){ return; }
//     uint32_t index = atomicAdd(nuniq,1);

// }

__global__ void
get_batch_ids(uint8_t* bids, uint64_t* edges, int* vptr, uint32_t E_uniq, uint32_t B){

    // -- get face index --
    uint32_t edge_index = threadIdx.x + blockIdx.x*blockDim.x;
    if (edge_index >= E_uniq){ return; }

    // -- read edges --
    uint64_t edge_pair = edges[edge_index];
    uint32_t e0 = static_cast<uint32_t>(edge_pair>>32);
    uint32_t e1 = static_cast<uint32_t>(edge_pair);
    // printf("(e0,e1): (%d, %d)\n",e0,e1);
    // assert(false);

    // -- infer batch index --
    int batch_index = 0;
    while(batch_index < B){
        if (e0 >= vptr[batch_index]){ break; }
        batch_index++;
    }
    assert(batch_index<B);

    // -- write batch index --
    bids[edge_index] = batch_index;

}



void manifold_edges(PointCloudData& data){

    // optionally copy... for now in-place

    //
    // Step 1: For Each Edge, Gather the Incident Faces
    //

    printf("step 1\n");
    auto [faces,unique_edge_keys,ptr,bids,counts] = gather_faces_by_edge(data);
    uint32_t E_uniq = unique_edge_keys.size();
    // printf("E_uniq, E: %d %d\n",E_uniq,data.E);
    int num_invalid = thrust::count_if(counts.begin(), counts.end(), [] __device__ (int c) { return c > 2; });
    if (num_invalid == 0){ 
        printf("No non-manifold edges.\n");
        return; 
    } // no non-manifold edges
    printf("num_invalid: %d\n",num_invalid);
    int NumThreads = 512;
    int UniqEdgeBlocks = ceil( double(E_uniq) / double(NumThreads));

    //
    // Step 2: Fix Faces and Update Features + Position
    //

    // -- number of new vertices --
    thrust::device_vector<uint32_t> new_num(data.B,0);
    count_num_new<<<UniqEdgeBlocks,NumThreads>>>(new_num.data().get(),bids.data().get(),unique_edge_keys.data().get(),counts.data().get(),E_uniq);
    thrust::device_vector<uint32_t> new_csum(data.B+1,0);
    thrust::inclusive_scan(new_num.begin(), new_num.end(), new_csum.begin() + 1);
    uint32_t new_num_total = new_csum[data.B];
    printf("num new total: %d\n",new_num_total);

    // -- update faces --
    thrust::device_vector<bool> mask(3*data.F);
    // thrust::device_vector<uint32_t> fmap(3*data.F);
    mark_faces_for_update<<<UniqEdgeBlocks,NumThreads>>>(mask.data().get(),//fmap.data().get(),
                                                        data.faces.data().get(),
                                                        faces.data().get(),ptr.data().get(),
                                                        unique_edge_keys.data().get(),
                                                        bids.data().get(),data.vptr.data().get(),data.fptr.data().get(),E_uniq);
    

    //
    //    Copy Original Data into Expanded Vectors
    //

    // -- copy vertex --
    uint32_t V_old = data.V;
    data.V = data.V + new_num_total;
    data.ftrs.resize(data.V); // explicit per-batch copy.
    data.pos.resize(data.V);
    data.vertex_batch_ids.resize(data.V); // explicit per-batch copy.
    {
        auto ftrs = data.ftrs;
        auto pos = data.pos;
        auto vertex_batch_ids = data.vertex_batch_ids;
        int VertexBlocks = ceil( double(V_old) / NumThreads );
        copy_vertex_info<<<VertexBlocks,NumThreads>>>(ftrs.data().get(),pos.data().get(),vertex_batch_ids.data().get(),
                                                    data.ftrs.data().get(),data.pos.data().get(),data.vertex_batch_ids.data().get(),
                                                    new_csum.data().get(),V_old);
        data.ftrs = ftrs;
        data.pos = pos;
        data.vertex_batch_ids = vertex_batch_ids;
    }

    
    // -- copy over (the same but differently ordered) uniq edges --
    // assert(E_uniq == E_old);
    // int EdgeBlocks = ceil( double(data.E) / NumThreads );
    // copy_over_edges<<<EdgeBlocks,NumThreads>>>(unique_edge_keys.data().get(),data.edges.data().get(),data.E);

    // -- copy edges --
    int E_old = data.E;
    data.E = data.E + 2*new_num_total; // 1 new vertex is 2 new edges
    data.edges.resize(2*data.E);
    data.edge_batch_ids.resize(data.E);
    {
        auto edges = data.edges;
        auto edge_batch_ids = data.edge_batch_ids;
        int EdgeBlocks = ceil( double(E_old) / NumThreads );
        copy_edge_info<<<EdgeBlocks,NumThreads>>>(edges.data().get(),edge_batch_ids.data().get(),
                                                  data.edges.data().get(),data.edge_batch_ids.data().get(),
                                                  new_csum.data().get(),E_old);
        data.edges = edges;
        data.edge_batch_ids = edge_batch_ids;
    }

    // -- update batch info --
    auto vptr = data.vptr;
    auto eptr = data.eptr;
    {
        int NumBlocks = ceil( double(data.B+1) / NumThreads );
        update_batch_info<<<NumBlocks,NumThreads>>>(data.vptr.data().get(),data.eptr.data().get(),new_csum.data().get(),data.B);
        int VertexBlocks = ceil( double(data.V) / NumThreads );
        fill_ve_batch_ids<<<VertexBlocks,NumThreads>>>(data.vertex_batch_ids.data().get(),data.vptr.data().get(),data.V,data.B);
        int EdgeBlocks = ceil( double(data.E) / NumThreads );
        fill_ve_batch_ids<<<EdgeBlocks,NumThreads>>>(data.edge_batch_ids.data().get(),data.eptr.data().get(),data.E,data.B);
    }
    // int _E_last = eptr[1];
    // printf("_E_last, data.E %d %d\n",_E_last,data.E);

    // -- we run this after reading scene from file; currently it won't work if other attributes computed --
    assert(data.csr_edges.empty());
    assert(data.csr_eptr.empty());
    assert(data.gcolors.empty());
    assert(data.face_labels.empty());
    assert(data.face_colors.empty());
    assert(data.face_vnames.empty());
    assert(data.labels.empty());
    assert(data.border.empty());

    // 
    //  Fill with New Vertex Copies and Edges
    //

    thrust::device_vector<uint32_t> Vnew(data.B,0);
    thrust::device_vector<uint32_t> Enew(data.B,0);
    int FaceBlocks = ceil( double(data.F) / double(NumThreads));
    update_faces_edges_verts<<<FaceBlocks,NumThreads>>>(data.ftrs.data().get(),data.pos.data().get(),
                                                        data.edges.data().get(),
                                                        data.faces.data().get(),data.face_batch_ids.data().get(),
                                                        mask.data().get(),//fmap.data().get(),
                                                        bids.data().get(),data.vptr.data().get(),data.eptr.data().get(),data.fptr.data().get(),
                                                        vptr.data().get(),eptr.data().get(),
                                                        Vnew.data().get(),Enew.data().get(),data.F);

    // -- check [for dev only] --
    {
        auto [_faces,_unique_edge_keys,_ptr,_bids,_counts] = gather_faces_by_edge(data);
        uint32_t E_uniq = _unique_edge_keys.size();
        int num_invalid_chk = thrust::count_if(_counts.begin(), _counts.end(), [] __device__ (int c) { return c > 2; });
        assert(num_invalid_chk == 0);
    }
    return;
}


std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint64_t>,thrust::device_vector<uint32_t>,thrust::device_vector<uint8_t>,thrust::device_vector<uint32_t>>
gather_faces_by_edge(PointCloudData& data){

    // -- launch params --
    int NumThreads = 512;
    int EdgeBlocks = ceil( double(data.E) / double(NumThreads));
    int FaceBlocks = ceil( double(data.F) / double(NumThreads));
    int AugFaceBlocks = ceil( double(3*data.F) / double(NumThreads));

    // -- encode edges --
    thrust::device_vector<uint64_t> edges(3*data.F);
    thrust::device_vector<uint64_t> faces(3*data.F);
    // uint64_t* edges_dptr = thrust::raw_pointer_cast(edges.data()); 
    // uint64_t* faces_dptr = thrust::raw_pointer_cast(faces.data()); 
    encode_edges_and_faces<<<FaceBlocks,NumThreads>>>(edges.data().get(),faces.data().get(),
                                                      data.faces_ptr(),data.face_batch_ids.data().get(),data.vptr.data().get(),data.fptr.data().get(),data.F);

    // -- sorted --
    thrust::device_vector<uint64_t> edges_sorted(3*data.F);
    thrust::device_vector<uint64_t> faces_sorted(3*data.F);
    // uint64_t* edges_sorted_dptr = thrust::raw_pointer_cast(edges_sorted.data()); 
    // uint64_t* faces_sorted_dptr = thrust::raw_pointer_cast(faces_sorted.data()); 

    // printf("pre sort.\n");
    // -- sort edges --
    {

        size_t temp_bytes = 0;

        // Determine temp storage size
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            edges.data().get(),edges_sorted.data().get(),
            faces.data().get(),faces_sorted.data().get(),
            3*data.F);

        // allocate temp storage
        thrust::device_vector<uint8_t> d_temp(temp_bytes);

        // Perform sorting
        cub::DeviceRadixSort::SortPairs(
            d_temp.data().get(), temp_bytes,
            edges.data().get(),edges_sorted.data().get(),
            faces.data().get(),faces_sorted.data().get(),
            3*data.F);
    }

    // cudaDeviceSynchronize();
    // printf("pre rle.\n");

    // -- get uniq --
    thrust::device_vector<uint64_t> unique_edge_keys(3*data.F);
    thrust::device_vector<uint32_t> counts(3*data.F); 
    thrust::device_vector<uint32_t> nuniq_tr(1,0); 
    {
        size_t temp_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(
            nullptr, temp_bytes,
            edges_sorted.data().get(),
            unique_edge_keys.data().get(),
            counts.data().get(),
            nuniq_tr.data().get(),3*data.F
        );

        // allocate temp storage
        thrust::device_vector<uint8_t> d_temp(temp_bytes);

        // actual call
        cub::DeviceRunLengthEncode::Encode(
            d_temp.data().get(), temp_bytes,
            edges_sorted.data().get(),
            unique_edge_keys.data().get(),
            counts.data().get(),
            nuniq_tr.data().get(),3*data.F
        );
    }

    int nuniq = nuniq_tr[0];
    unique_edge_keys.resize(nuniq);
    counts.resize(nuniq);
    // cudaDeviceSynchronize();
    // printf("post rle.\n");

    // -- get pointer for access by uniq edge --
    thrust::device_vector<uint32_t> ptr(nuniq+1,0);
    thrust::inclusive_scan(counts.begin(), counts.end(), ptr.begin() + 1);

    // can i get the batch inds..?
    // -- get batch ids --
    thrust::device_vector<uint8_t> bids(nuniq);
    int UniqEdgeBlocks = ceil( double(nuniq) / NumThreads);
    get_batch_ids<<<UniqEdgeBlocks,NumThreads>>>(bids.data().get(),unique_edge_keys.data().get(),
                                                 data.vptr.data().get(), nuniq, data.B);

    return std::tuple(faces_sorted,unique_edge_keys,ptr,bids,counts);
}