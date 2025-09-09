

#include "structs_3d.h"

#define THREADS_PER_BLOCK 512


__global__ void
initial_interpolation(float3* ftrs, float3* pos, uint32_t* dual_faces, uint32_t* face_counts, 
                        const float3* in_ftrs, const float3* in_pos, const uint32_t* faces, const uint32_t* csr_eptr, uint32_t F){
	
    // -- get face index --
    uint32_t face_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (face_index>=F) return;

    // -- unpack triangle --
    uint32_t v0 = faces[3*face_index+0];
    uint32_t v1 = faces[3*face_index+1];
    uint32_t v2 = faces[3*face_index+2];

    // -- get vertex info --
    float3 pos0 = in_pos[v0];
    float3 ftrs0 = in_ftrs[v0];
    float3 pos1 = in_pos[v1];
    float3 ftrs1 = in_ftrs[v1];
    float3 pos2 = in_pos[v2];
    float3 ftrs2 = in_ftrs[v2];

    // -- equally weighted interpolation --
    float3 pos_f;
    pos_f.x = (pos0.x + pos1.x + pos2.x)/3;
    pos_f.y = (pos0.y + pos1.y + pos2.y)/3;
    pos_f.z = (pos0.z + pos1.z + pos2.z)/3;

    float3 ftrs_f;
    ftrs_f.x = (ftrs0.x + ftrs1.x + ftrs2.x)/3;
    ftrs_f.y = (ftrs0.y + ftrs1.y + ftrs2.y)/3;
    ftrs_f.z = (ftrs0.z + ftrs1.z + ftrs2.z)/3;

    // -- write --
    pos[face_index] = pos_f;
    ftrs[face_index] = ftrs_f;

    // // -- neighbors --
    // edges[6*face_index+0] = min(v0,v1);
    // edges[6*face_index+1] = max(v0,v1);
    // edges[6*face_index+2] = min(v1,v2);
    // edges[6*face_index+3] = max(v1,v2);
    // edges[6*face_index+4] = min(v2,v0);
    // edges[6*face_index+5] = max(v2,v0);

    // -- the number of faces that include the particular vertex --
    uint32_t i0 = atomicAdd(&face_counts[v0],1);
    uint32_t i1 = atomicAdd(&face_counts[v1],1);
    uint32_t i2 = atomicAdd(&face_counts[v2],1);

    // -- gather all the faces for each vertex --
    uint32_t s0 = csr_eptr[v0];
    uint32_t s0_end = csr_eptr[v0+1];
    uint32_t s1 = csr_eptr[v1];
    uint32_t s1_end = csr_eptr[v1+1];
    uint32_t s2 = csr_eptr[v2];
    uint32_t s2_end = csr_eptr[v2+1];
    // if((s0+i0) >= s0_end){
    //     printf("v0,face: %d %d | %d %d %d\n",v0,face_index,i0,s0,s0_end);
    // }
    assert((s0+i0) < s0_end);
    assert((s1+i1) < s1_end);
    assert((s2+i2) < s2_end);
    dual_faces[s0+i0] = face_index;
    dual_faces[s1+i1] = face_index;
    dual_faces[s2+i2] = face_index;

}


// __global__ void
// initial_faces(uint32_t* dual_faces, uint32_t* face_counts, uint32_t* face_eptr, const uint32_t* faces, uint32_t F){

//     // -- get face index --
//     uint32_t face_index = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (face_index>=F) return;

//     // -- unpack triangle --
//     uint32_t v0 = faces[3*face_index+0];
//     uint32_t v1 = faces[3*face_index+1];
//     uint32_t v2 = faces[3*face_index+2];

//     // -- the number of faces that include the particular vertex --
//     uint32_t i0 = atomicAdd(&face_counts[v0],1);
//     uint32_t i1 = atomicAdd(&face_counts[v1],1);
//     uint32_t i2 = atomicAdd(&face_counts[v2],1);

//     // -- gather all the faces for each vertex --
//     uint32_t s0 = face_eptr[v0];
//     uint32_t s0_end = face_eptr[v0+1];
//     uint32_t s1 = face_eptr[v1];
//     uint32_t s1_end = face_eptr[v1+1];
//     uint32_t s2 = face_eptr[v2];
//     uint32_t s2_end = face_eptr[v2+1];
//     // if((s0+i0) >= s0_end){
//     //     printf("v0,face: %d %d\n",v0,face_index,s0);
//     // }
//     assert((s0+i0) < s0_end);
//     assert((s1+i1) < s1_end);
//     assert((s2+i2) < s2_end);
//     dual_faces[s0+i0] = face_index;
//     dual_faces[s1+i1] = face_index;
//     dual_faces[s2+i2] = face_index;
// }
	

__global__ void
place_faces(uint32_t* edges, uint32_t* count, const uint32_t* primal_faces, const uint32_t* csr_edges, const uint32_t* csr_eptr, uint32_t F, uint32_t V){


    // -- get face index --
    uint32_t face_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (face_index>=F) return;

    bool success = false;
    bool any = false;
    for(int ix=0; ix < 3; ix++){

        success = false;
        any = false;
        uint32_t v0 = primal_faces[3*face_index+ix];
        uint32_t v1 = primal_faces[3*face_index+((ix + 1) % 3)];
        assert(v0 != v1);
        uint32_t jx0 = 0;
        // assert(v0 != v1);

        // bool conda = (v0 == 12372) && (v1 == 81382);
        // bool condb = (v1 == 12372) && (v0 == 81382);
        // bool conda = (v0 == 81399) || (v1 == 81399);
        // bool condb = false;
        // bool cond = conda || condb;
        // if (conda || condb){
        //     printf("face_index, v0, v1: %d %d %d\n",face_index, v0, v1);
        // }

        // if (cond){
        //     printf("(%d,%d): csr_eptr[v0], csr_eptr[v0+1], csr_eptr[v1], csr_eptr[v1+1] (%d,%d,%d,%d)\n",v0,v1,csr_eptr[v0], csr_eptr[v0+1], csr_eptr[v1], csr_eptr[v1+1]);
        // }
        uint32_t start = csr_eptr[v0];
        uint32_t end = csr_eptr[v0+1];
        for(int jx = start; jx < end; jx++){
            //if(cond){ printf("(%d,%d) [%d] csr_edges[%d]: %d\n",v0,v1,v0,jx,csr_edges[jx]);}
            if(csr_edges[jx] == v1){
                jx0 = jx;
                any = true;
                uint32_t old_label = atomicCAS(
                    (unsigned int*)&edges[jx],
                    UINT32_MAX, (unsigned int)face_index
                );
                success = old_label==UINT32_MAX;
                //if (success){ atomicAdd(&count[jx],1); }
                int old = atomicAdd(&count[jx],1);
                // if (old >= 2){
                //     printf("(%d,%d) %d\n",v0,v1,old);
                // } 
                break;
            }
        }

        // -- don't do this loop if not needed --
        if (success){ continue; }

        any = false;
        start = csr_eptr[v1];
        end = csr_eptr[v1+1];
        for(int jx = start; jx < end; jx++){
            if(csr_edges[jx] == v0){
                //if(cond){ printf("[%d] csr_edges[%d]: %d\n",v1,jx,csr_edges[jx]);}
                any = true;
                uint32_t old_label = atomicCAS(
                    (unsigned int*)&edges[jx],
                    UINT32_MAX, (unsigned int)face_index
                );
                success = old_label==UINT32_MAX;
                //if (success){ atomicAdd(&count[jx],1); }
                // if(!success){

                //     uint32_t v0_0 = primal_faces[3*face_index+0];
                //     uint32_t v1_0 = primal_faces[3*face_index+1];
                //     uint32_t v2_0 = primal_faces[3*face_index+2];

                //     uint32_t v0_1 = primal_faces[3*old_label+0];
                //     uint32_t v1_1 = primal_faces[3*old_label+1];
                //     uint32_t v2_1 = primal_faces[3*old_label+2];

                //     uint32_t v0_2 = primal_faces[3*edges[jx0]+0];
                //     uint32_t v1_2 = primal_faces[3*edges[jx0]+1];
                //     uint32_t v2_2 = primal_faces[3*edges[jx0]+2];
                //     printf("(%d,%d) face[%d]: (%d,%d,%d) face[%d]: (%d,%d,%d) face[%d]: (%d,%d,%d)\n",
                //         v0,v1,
                //         face_index,v0_0,v1_0,v2_0,
                //         old_label,v0_1,v1_1,v2_1,
                //         edges[jx0],v0_2,v1_2,v2_2);
                // }
                int old = atomicAdd(&count[jx],1);
                // if (old >= 2){
                //     printf("(%d,%d) %d\n",v1,v0,old);
                // } 
                break;
            }
        }
        if(!any){
            //printf("v0,v1: (%d,%d)\n",v0,v1);

            // printf("neigh v0:");
            // start = csr_eptr[v0];
            // end = csr_eptr[v0+1];
            // for(int jx = start; jx < end; jx++){
            //     printf(" %d",csr_edges[jx]);
            // }
            // printf("\n");

            // printf("neigh v1:");
            // start = csr_eptr[v1];
            // end = csr_eptr[v1+1];
            // for(int jx = start; jx < end; jx++){
            //     printf(" %d",csr_edges[jx]);
            // }
            // printf("\n");
        }
        //assert(success && any);
        assert(any);

    }

}

__global__ void
get_face_csr(uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t* E, bool* flag, bool* eflag,
            const uint32_t* primal_faces, const uint32_t* faces, const uint32_t* pr_csr_edges, const uint32_t* pr_csr_eptr, uint32_t F){

    // the goal is to get the edges of each face.
    // for a triangle mesh, an (dual) edge appears between two (primal) faces when they share a border (a primal edge)
    //
    //            v0 - v3 
    //           /  \  / 
    //          v1 - v2
    //
    //  These faces share (v0,v2) so the two faces have an "edge" 
    //
    //  The problem is that we write all the edges initially, so don't need (v0,v1)... but then we don't find an alt face.. so?
    //
    //
    // for each edge, we wrote the two face ids at each of the (v0,v1) or (v1,v0) locations in the csr_edges for easy of access.
    // now we just have to check both and read the one that doesn't match us
    // its possible only one edge borders a pair, so we skip that one.
    //
    //

    // can't we have some edges appear once? -- yes!
    // but then how is it that a face is written at vertex (v0,v1)? edge[v0,i(v1)] = f0 but not edge[v1,i(v0)] = f1?
    // hmm.... this actually seems fine... since some edges at the boundary...
    // 
    // then i don't think there is any "x2" idea here... but there is right? since we have edge[f0,i(f1)] = f1 and edge[f1,i(f0)] = f1
    // but then how do we get an odd number?
    // ...
    // 


    // -- get face index --
    uint32_t face_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (face_index>=F) return;

    int edge_count = 0;
    bool success = false;
    uint32_t read_face_index = UINT32_MAX;
    for(int ix=0; ix < 3; ix++){

        success = false;
        read_face_index = UINT32_MAX;

        uint32_t v0 = primal_faces[3*face_index+ix];
        uint32_t v1 = primal_faces[3*face_index+((ix + 1) % 3)];

        if (v0 == UINT32_MAX){ continue; }

        uint32_t start = pr_csr_eptr[v0];
        uint32_t end = pr_csr_eptr[v0+1];
        for(int jx = start; jx < end; jx++){
            if(pr_csr_edges[jx] == v1){
                read_face_index = faces[jx];
                success = (face_index!=read_face_index) && (read_face_index != UINT32_MAX);
                if (success){
                    csr_edges[3*face_index+ix] = read_face_index;
                    flag[3*face_index+ix] = true;
                    edge_count++;
                }
            }
        }

        // -- don't do this loop if not needed --
        if (success){ continue; }
        start = pr_csr_eptr[v1];
        end = pr_csr_eptr[v1+1];
        for(int jx = start; jx < end; jx++){
            if(pr_csr_edges[jx] == v0){
                read_face_index = faces[jx];
                success = (face_index!=read_face_index) && (read_face_index != UINT32_MAX);
                if (success){
                    csr_edges[3*face_index+ix] = read_face_index;
                    flag[3*face_index+ix] = true;
                    edge_count++;
                }
            }
        }

        //assert(success);

    }

    csr_eptr[face_index] = edge_count;
    eflag[face_index] = edge_count>0; // thinking; keep everyone? otherwise, i think we need to relabel... maybe not since csr_eptr catches this...
    //assert(edge_count>0);
    atomicAdd(E,edge_count);

}

__global__ void
get_uniq_edges(uint32_t* edges, uint32_t* write_offset, const uint32_t* csr_edges, const uint32_t* csr_eptr, uint32_t V){

    // -- get vertex index --
    uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
	if (vertex>=V) return;
    uint32_t neighs[3] = {UINT32_MAX}; // max degree is 3

    // -- .. --
    uint32_t start = csr_eptr[vertex];
    uint32_t end = csr_eptr[vertex+1];

    // -- check --
    // if (vertex == 563800){
    //     printf("start end: %d %d\n",start,end);
    // }

    assert ((end - start) <= 3);
    int num = 0;
    for(uint32_t index = start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        // if ((vertex > 0) && (neigh == 0)){
        //     printf("[%d]: %d\n",vertex,neigh);
        // }

        if (vertex >= neigh){ continue; }
        neighs[num] = neigh;
        num++;
    }

    // -- return early if elidgable --
    if (num == 0){ return; }

    // -- get place to write --
    uint32_t write_index = atomicAdd(write_offset,num);

    // -- write neighbors --
    for (uint32_t index = 0; index < num; index++){
        edges[2*(write_index+index)+0] = vertex;
        edges[2*(write_index+index)+1] = neighs[index];
    }

}

// __global__ void
// init_faces(uint32_t* faces, uint32_t* face_counts, uint32_t* csr_edges, uint32_t* csr_eptr, uint32_t* primal_csr_eptr, uint32_t F){

//     // -- get face index --
//     uint32_t face = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (face>=F) return;

//     // -- start --
//     uint32_t start = csr_edges[face];
//     uint32_t end = csr_edges[face+1];
//     uint32_t size = end - start;
//     if (size == 0){ return; }
//     for (int index = start; index < end; index++){
//         uint32_t neigh = csr_edges[index];

//     }


//     // // -- the number of faces that include the particular vertex --
//     // uint32_t i0 = atomicAdd(&face_counts[v0],1);
//     // uint32_t i1 = atomicAdd(&face_counts[v1],1);
//     // uint32_t i2 = atomicAdd(&face_counts[v2],1);

//     // // -- gather all the faces for each vertex --
//     // uint32_t s0 = csr_eptr[v0];
//     // uint32_t s1 = csr_eptr[v1];
//     // uint32_t s2 = csr_eptr[v2];
//     // dual_faces[s0+i0] = face_index;
//     // dual_faces[s1+i1] = face_index;
//     // dual_faces[s2+i2] = face_index;
// }

__global__ void
construct_dual_faces(uint32_t* faces, uint32_t* faces_size, bool* flag, bool* face_flag, uint32_t* F, const uint32_t* dual_csr_edges, const uint32_t* dual_csr_eptr,  const uint32_t* csr_eptr, uint32_t V){

    // -- get primal vertex --
    uint32_t primal_vertex = threadIdx.x + blockIdx.x * blockDim.x;
	if (primal_vertex>=V) return;

    // -- allocate --
    constexpr int MAX_NEIGHS = 32; // max degree of primal
    uint32_t neighs[MAX_NEIGHS] = {UINT32_MAX};
    #pragma unroll
    for (int i = 0; i < MAX_NEIGHS; i++) {
        neighs[i] = UINT32_MAX;
    }

    // -- get endpoints --
    uint32_t start = csr_eptr[primal_vertex];
    uint32_t end = csr_eptr[primal_vertex+1];
    int poly_size = end - start;
    //printf("poly_size: %d\n",poly_size);
    if (poly_size == 0){ return ;}
    assert(poly_size < MAX_NEIGHS);

    // -- read to memory --
    int num = 0;
    for (int index = start; index < end; index++){
        neighs[num] = faces[index];
        if (neighs[num] == UINT32_MAX){ break; }
        num++;
    }
    neighs[num] = neighs[0];
    // printf("neighs[num+1]: %d",neighs[num+1]);

    // -- info --
    // bool any_found = false;
    // for (int index=0; index < num; index++){
    //     bool conda = neighs[index] == 0;
    //     bool condb = neighs[index] == 78188;
    //     any_found = conda || condb;
    // }

    // for (int index=0; index < num; index++){
    //     if(any_found){
    //         printf("[%d] neigh[%d]: %d\n",primal_vertex,index,neighs[index]);
    //     }
    // }

     
    // -- path traversal --
    bool success;
    int selected_ix;
    for(int ix = 0; ix < num; ix++){
        
        success = false;
        selected_ix = -1;
        uint32_t face = neighs[ix];
        uint32_t prev = (ix > 0) ? neighs[ix-1] : UINT32_MAX;
        uint32_t start = dual_csr_eptr[face];
        uint32_t end = dual_csr_eptr[face+1];

        // -- Find a Neighbors of Vertex "face" that is Part of the Shuffled Polygon --
        for (uint32_t index = start; index < end; index++){
            uint32_t neigh = dual_csr_edges[index];
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
        if (!success){ return; }
        int tmp = neighs[selected_ix];
        neighs[selected_ix] = neighs[ix+1];
        neighs[ix+1] = tmp; 
        
    }
    
    // -- if bool is not set, it failed to close the loop --
    if (!success){ 
        return; 
    }
    atomicAdd(F,1);
    

    // -- info --
    // int init_num = num;
    // start = csr_eptr[primal_vertex];
    // end = csr_eptr[primal_vertex+1];
    // num = 0;
    // bool any_found = false;
    // for (int index = start; index < end; index++){
    //     if (neighs[num+1] == UINT32_MAX){ break; }
    //     bool conda = neighs[num] == 0;
    //     bool condb = false;
    //     any_found = conda || condb;
    //     num++;
    // }
    // num = 0;
    // for (int index = start; index < end; index++){
    //     if(any_found){
    //         printf("[%d,%d] neigh[%d]: %d\n",primal_vertex,init_num,index,neighs[num]);
    //     }
    //     num++;
    // }

    // -- mark for keeping and write if successful --
    num = 0;
    start = csr_eptr[primal_vertex];
    end = csr_eptr[primal_vertex+1];
    //end = start + num;
    for (int index = start; index < end; index++){
        if (neighs[num+1] == UINT32_MAX){ break; }
        faces[index] = neighs[num];
        flag[index] = true;
        num++;
    }
    faces_size[primal_vertex] = num;
    face_flag[primal_vertex] = true;
}


// todo: modify to allow for batching...
PointCloudData create_dual_mesh(PointCloudData& data){
    
    // -- allocate initial features --
    thrust::device_vector<float3> ftrs(data.F);
    float3* ftrs_dptr = thrust::raw_pointer_cast(ftrs.data());
    thrust::device_vector<float3> pos(data.F);
    float3* pos_dptr = thrust::raw_pointer_cast(pos.data());
    thrust::device_vector<uint32_t> faces(2*data.E,UINT32_MAX); // at most all edge pairs make-up the faces...?
    uint32_t* faces_dptr = thrust::raw_pointer_cast(faces.data());

    // -- simple fixed quantites --
    uint32_t B = data.B;
    uint32_t V = data.F;

    // -- some launch parameters --
    int NumThreads = THREADS_PER_BLOCK;
    int FaceBlocks = ceil( double(data.F) / double(NumThreads) ); 
    int VertexBlocks = ceil( double(data.V) / double(NumThreads) ); 

    //
    //
    //  Part 1: Get Vertices of Dual Graph
    //
    //

    // -- interpolate features, position, and read all edges --
    // thrust::device_vector<int> write_offsets(data.F,0);
    // int* wo_dptr = thrust::raw_pointer_cast(write_offsets.data());
    // thrust::device_vector<uint32_t> face_edge_pairs(6*data.F,UINT32_MAX);
    // uint32_t* fe_dptr = thrust::raw_pointer_cast(face_edge_pairs.data());
    thrust::device_vector<uint32_t> face_counts(data.V,0); // # faces a vertex touches
    uint32_t* fc_dptr = thrust::raw_pointer_cast(face_counts.data());
    initial_interpolation<<<FaceBlocks,NumThreads>>>(ftrs_dptr, pos_dptr, faces_dptr, fc_dptr, 
                                                    data.ftrs_ptr(), data.pos_ptr(), data.faces_ptr(), data.csr_eptr_ptr(), data.F);


    // -- get faces eptr --
    // thrust::device_vector<uint32_t> vfaces_eptr(data.V+1,0); // (at most) one dual face per primal vertex
    // thrust::inclusive_scan(face_counts.begin(), face_counts.end(), vfaces_eptr.begin() + 1);
    // thrust::fill(face_counts.begin(), face_counts.end(), 0);
    // uint32_t* vfaces_eptr_dptr = thrust::raw_pointer_cast(vfaces_eptr.data());
    // initial_faces<<<FaceBlocks,NumThreads>>>(faces_dptr, fc_dptr, vfaces_eptr_dptr,  data.faces_ptr(), data.F);
    uint32_t max_face_count = *thrust::max_element(face_counts.begin(), face_counts.end());
    printf("max_face_count: %d\n",max_face_count);


    //
    //
    //   Part 2: Get Edges of Dual Graph
    //
    //

    // -- places faces at each edge pair --
    // thrust::device_vector<int> write_offsets_v2(data.E,0);
    // int* wov2_dptr = thrust::raw_pointer_cast(write_offsets_v2.data());
    thrust::device_vector<uint32_t> edges_to_faces(2*data.E,UINT32_MAX); // same size as csr_edges
    uint32_t* e2f_dptr = thrust::raw_pointer_cast(edges_to_faces.data());
    thrust::device_vector<uint32_t> ecounts(2*data.E,0);
    uint32_t* ecounts_dptr = thrust::raw_pointer_cast(ecounts.data());
    place_faces<<<FaceBlocks,NumThreads>>>(e2f_dptr, ecounts_dptr, data.faces_ptr(), data.csr_edges_ptr(),  data.csr_eptr_ptr(), data.F, data.V);
    int max_count = *thrust::max_element(ecounts.begin(), ecounts.end());
    printf("max_count: %d\n",max_count);
    if (max_count > 2){
        printf("ERROR: mesh is non-manifold. Please modify this source code or manually clean-up the input mesh.\n");
        exit(1);
    }
        
    // -- get face csr-style edges --
    thrust::device_vector<uint32_t> csr_edges(3*data.F,UINT32_MAX); // I think should be 2*data.E but oh well...
    thrust::device_vector<uint32_t> edge_counts(data.F,0);
    thrust::device_vector<uint32_t> numEdges(1, 0); 
    thrust::device_vector<bool> flags(3*data.F, 0);
    thrust::device_vector<bool> eflags(data.F, 0);
    uint32_t* csr_edges_dptr = thrust::raw_pointer_cast(csr_edges.data());
    uint32_t* edge_counts_dptr = thrust::raw_pointer_cast(edge_counts.data());
    uint32_t* E_gpu = thrust::raw_pointer_cast(numEdges.data());
    bool* flags_dptr = thrust::raw_pointer_cast(flags.data());
    bool* eflags_dptr = thrust::raw_pointer_cast(eflags.data());
    get_face_csr<<<FaceBlocks,NumThreads>>>(csr_edges_dptr, edge_counts_dptr, E_gpu, flags_dptr, eflags_dptr, data.faces_ptr(), e2f_dptr, data.csr_edges_ptr(),  data.csr_eptr_ptr(), data.F);
    //int eflags_sum = thrust::reduce(eflags.begin(), eflags.end(), 0, thrust::plus<int>());
    int eflags_sum = thrust::count(eflags.begin(), eflags.end(), true);
    uint32_t E_tmp = numEdges[0];
    //printf("E_tmp: %d\n",E_tmp);
    printf("eflags_sum: %d\n",eflags_sum);
    assert(E_tmp % 2 == 0); // must be even
    uint32_t E = E_tmp/2;

    cudaDeviceSynchronize();

    // -- Compactify Edges --
    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        thrust::device_vector<uint32_t> dnum(1, 0); 
        unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            csr_edges_dptr, flags_dptr,
            d_num_selected, 3*data.F);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            csr_edges_dptr, flags_dptr,
            d_num_selected, 3*data.F);
        uint32_t _E = dnum[0];
        csr_edges.resize(_E);
        printf("E,_E: %d %d\n",E_tmp,_E);
        assert(E_tmp == _E);
        cudaFree(d_temp);
    }
    
    //cudaDeviceSynchronize();
    // -- Compactify Counts a BAD idea; this requires re-labeling the face ids but we don't want to do this if (yet) --
    // uint32_t _V;
    // {
    //     void* d_temp = nullptr;
    //     size_t temp_bytes = 0;
    //     thrust::device_vector<uint32_t> dnum(1, 0); 
    //     unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
    //     cub::DeviceSelect::Flagged(
    //         d_temp, temp_bytes,
    //         edge_counts_dptr, eflags_dptr,
    //         d_num_selected, data.F);
    //     cudaMalloc(&d_temp, temp_bytes);
    //     cub::DeviceSelect::Flagged(
    //         d_temp, temp_bytes,
    //         edge_counts_dptr, eflags_dptr,
    //         d_num_selected, data.F);
    //     cudaFree(d_temp);
    //     _V = dnum[0];
    // }
    // exit(1);

    // -- make the csr_edges actually accumulated --
    thrust::device_vector<uint32_t> csr_eptr(data.F+1,0);
    thrust::inclusive_scan(edge_counts.begin(), edge_counts.end(), csr_eptr.begin() + 1);
    uint32_t* csr_eptr_dptr = thrust::raw_pointer_cast(csr_eptr.data());
    uint32_t fsize = csr_eptr[data.F];

    // -- get unique edge pairs --
    thrust::device_vector<uint32_t> edges(2*E,UINT32_MAX);
    thrust::device_vector<uint32_t> write_offset_v2(1,0);
    uint32_t* edges_dptr = thrust::raw_pointer_cast(edges.data());
    uint32_t* wov2_dptr = thrust::raw_pointer_cast(write_offset_v2.data());
    get_uniq_edges<<<FaceBlocks,NumThreads>>>(edges_dptr, wov2_dptr, csr_edges_dptr, csr_eptr_dptr, data.F);


    //
    //
    //   Part 3: Get Faces of Dual Graph
    //
    //

    // fill the "faces" with only the valid edge-pairs according to the DUAL csr edges
    //init_faces<<<FaceBlocks,NumThreads>>>(faces_dptr,fc_dptr,csr_edges_dptr, csr_eptr_dptr, data.F);

    // Using all the pairs from the faces includes edges that ARE NOT edges in the dual graph.
    // Thus, we should only reorder dual_csr_edges rather than the faces.

    // -- get faces --
    printf("fsize , 2*E, 2*data.E: %d %d %d\n",fsize,2*E,2*data.E); // same! but different from 2*data.E
    thrust::device_vector<bool> face_flag(2*data.E,0); // upper bounded by 2 x # unique edge pairs
    thrust::device_vector<bool> face_size_flag(data.V,0); // upper bounded by 2 x # unique edge pairs
    bool* face_flag_dptr = thrust::raw_pointer_cast(face_flag.data());
    bool* face_size_flag_dptr = thrust::raw_pointer_cast(face_size_flag.data());
    thrust::device_vector<uint32_t> nfaces(1,0);
    uint32_t* nfaces_dptr = thrust::raw_pointer_cast(nfaces.data());
    thrust::device_vector<uint32_t> faces_size(data.V,0);
    uint32_t* fsize_dptr = thrust::raw_pointer_cast(faces_size.data());
    //construct_dual_faces<<<VertexBlocks,NumThreads>>>(faces_dptr,fsize_dptr,face_flag_dptr,face_size_flag_dptr,nfaces_dptr,csr_edges_dptr,csr_eptr_dptr,data.csr_eptr_ptr(),data.V);
    construct_dual_faces<<<VertexBlocks,NumThreads>>>(faces_dptr,fsize_dptr,face_flag_dptr,face_size_flag_dptr,nfaces_dptr,csr_edges_dptr,csr_eptr_dptr,data.csr_eptr_ptr(),data.V);
    uint32_t F = nfaces[0];
    printf("F,_F: %d,%d\n",F,data.F);

    // -- names for relabeling --
    thrust::device_vector<uint32_t> vnames(data.V);
    thrust::sequence(vnames.begin(), vnames.end(), 0);
    uint32_t* vname_dptr = thrust::raw_pointer_cast(vnames.data());

    // -- Compactify Faces IDS --
    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        thrust::device_vector<uint32_t> dnum(1, 0); 
        unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            faces_dptr, face_flag_dptr,
            d_num_selected, 2*data.E);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            faces_dptr, face_flag_dptr,
            d_num_selected, 2*data.E);
        cudaFree(d_temp);
    }

    // -- Compactify Size of Each Face --
    uint32_t v2_F = 0;
    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        thrust::device_vector<uint32_t> dnum(1, 0); 
        unsigned int* d_num_selected = thrust::raw_pointer_cast(dnum.data());
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            fsize_dptr, face_size_flag_dptr,
            d_num_selected, data.V);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            fsize_dptr, face_size_flag_dptr,
            d_num_selected, data.V);
        cub::DeviceSelect::Flagged(
            d_temp, temp_bytes,
            vname_dptr, face_size_flag_dptr,
            d_num_selected, data.V);
        cudaFree(d_temp);
        v2_F = dnum[0];
        assert(v2_F == F);
    }
 
    // -- get offsets per vertex... kind of weird... _nfaces ~= 230k vs F ~= 40k --
    vnames.resize(F);
    faces_size.resize(F);
    thrust::device_vector<uint32_t> faces_csum(F+1, 0);
    thrust::inclusive_scan(faces_size.begin(), faces_size.end(), faces_csum.begin() + 1);
    uint32_t _nfaces = faces_csum[F];
    printf("F, _nfaces: %d %d\n",F,_nfaces);
    // assert(F == _nfaces); // sanity check

    // -- silly for now; mostly a lie --
    thrust::device_vector<uint8_t> vertex_batch_ids(data.F,0);
    thrust::device_vector<uint8_t> edge_batch_ids(E,0);
    thrust::device_vector<uint8_t> face_batch_ids(F,0);
    thrust::device_vector<int> vptr(B+1,0);
    thrust::device_vector<int> eptr(B+1,0);
    thrust::device_vector<int> fptr(B+1,0);
    vptr[B] = V;
    eptr[B] = E;
    fptr[B] = F;

    // todo: allow for device_vectors in constructor
    PointCloudData dual{
        ftrs,pos,faces,faces_csum,edges,
        vertex_batch_ids,edge_batch_ids,face_batch_ids,
        vptr,eptr,fptr,
        data.bounding_boxes,B,V,E,F};
    //dual.faces_eptr = std::move(faces_csum);
    dual.csr_edges = std::move(csr_edges);
    dual.csr_eptr = std::move(csr_eptr);
    dual.face_vnames = std::move(vnames);
    // dual.edges.resize(0);
    // dual.E = 0;
    return dual;

}
