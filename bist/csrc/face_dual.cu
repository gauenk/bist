

#include "structs_3d.h"


PointCloudData create_dual_mesh(PointCloudData& data){

    thrust::device_vector<float3> ftrs;
    thrust::device_vector<float3> pos;
    thrust::device_vector<uint32_t> faces;
    thrust::device_vector<uint32_t> edges;
    thrust::device_vector<uint32_t> csr_edges;
    thrust::device_vector<uint32_t> csr_eptr;
    thrust::device_vector<uint8_t> vertex_batch_ids;
    thrust::device_vector<uint8_t> edge_batch_ids;
    thrust::device_vector<int> vptr;
    thrust::device_vector<int> eptr;
    thrust::device_vector<int> fptr;
    
    int B = data.B;
    int V = data.V;
    int E = data.E;
    int F = data.F;

    // todo: allow for device_vectors in constructor
    PointCloudData dual{
        ftrs,pos,faces,edges,
        vertex_batch_ids,edge_batch_ids,
        vptr,eptr,fptr,
        data.bounding_boxes,B,V,E,F};
    return dual;

}
