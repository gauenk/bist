
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512

#include <assert.h>
#include <stdio.h>
#include "structs_3d.h"
#include "seg_utils_3d.h"
#include "update_seg_3d.h"
#include "articulation_points.h"

/**********************************************


         Segmentation Update Kernel


**********************************************/


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* AND are not adjacent to each other
*/



/**********************************************

              Main Function

**********************************************/

__host__ void update_seg(spix_params* aos_params, spix_helper* sp_helper, PointCloudData& data, SuperpixelParams3d& soa_params, SpixMetaData& args, Logger* logger){

    // -- unpack pointers --
    uint32_t* spix = soa_params.spix_ptr();
    bool* border = soa_params.border_ptr();
    bool* is_simple_point = soa_params.is_simple_point_ptr();
    uint8_t* neigh_neq = soa_params.neigh_neq_ptr();

    int NumThreads = THREADS_PER_BLOCK;
    int VertexBlocks = ceil( double(data.V) / double(THREADS_PER_BLOCK) ); 

    int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_WARP = 32;
    int threads_per_block = WARPS_PER_BLOCK * THREADS_PER_WARP;
    int ArtNumThreads(threads_per_block);
    int ArtNumBlocks((data.V + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    cudaMemset(is_simple_point, 1, data.V*sizeof(bool));
        
    // // -- [dev only!] --
    // cudaMemset(border, 0, data.V*sizeof(bool));
    // find_border_vertex<<<VertexBlocks,NumThreads>>>(spix,border,data.csr_edges,data.csr_eptr,data.V);
    // if (logger!=nullptr){
    //     logger->boundary_update(data,soa_params,aos_params);
    // }
    // return;
    printf("data.gchrome: %d\n",data.gchrome);

    // assert(nbatch==1);
    for (int iter = 0 ; iter < args.niters_seg; iter++){
        cudaMemset(border, 0, data.V*sizeof(bool));
        find_border_vertex<<<VertexBlocks,NumThreads>>>(spix,border,data.csr_edges_ptr(),data.csr_eptr_ptr(),data.V);

        for (int graph_color_index=0; graph_color_index < data.gchrome; graph_color_index++){

            //approximate_articulation_points<<<ArtNumBlocks,ArtNumThreads>>>(spix,data.csr_edges_ptr(),data.csr_eptr_ptr(),is_simple_point,neigh_neq,data.V);
            if (args.use_dual){
                //approximate_articulation_points_v0<<<VertexBlocks,NumThreads>>>(spix,border,data.pos_ptr(),data.csr_edges_ptr(),data.csr_eptr_ptr(),is_simple_point,neigh_neq,data.gcolors_ptr(),graph_color_index,data.V);
            }
            //approximate_articulation_points_v1<<<VertexBlocks,NumThreads>>>(spix,border,data.csr_edges_ptr(),data.csr_eptr_ptr(),is_simple_point,neigh_neq,data.gcolors_ptr(),graph_color_index,data.V);
            // gpuErrchk( cudaPeekAtLastError() );
            // gpuErrchk( cudaDeviceSynchronize() );
            //printf("step.\n");

            update_seg_subset<<<VertexBlocks,NumThreads>>>(aos_params,spix,border,is_simple_point,
                                                          data.ftrs_ptr(),data.pos_ptr(),neigh_neq,
                                                          data.gcolors_ptr(),graph_color_index,
                                                          data.csr_edges_ptr(),data.csr_eptr_ptr(),
                                                          data.V,args.sigma2_app,args.potts);
            // gpuErrchk( cudaPeekAtLastError() );
            // gpuErrchk( cudaDeviceSynchronize() );

        }
    }
            
    // -- log it! --
    if (logger!=nullptr){
        logger->boundary_update(data,soa_params,aos_params);
    }
}

__global__
void update_seg_subset(spix_params* params, uint32_t* spix, 
                        const bool* border, const bool* is_simple,
                        const float3* ftrs, const float3* pos,
                        const uint8_t* neigh_neq, const uint8_t* gcolors, const int gcolor, 
                        const uint32_t* csr_edges, const uint32_t* csr_ptr, 
                        const uint32_t V, const float sigma2_app, const float potts_coeff){

    // -- get current vertex --
    int vertex = threadIdx.x + blockIdx.x*blockDim.x; 
    if (vertex>=V)  return;

    // -- check if change is possible [~ believe in yourself, vertex! ~?] --
    int is_colored = gcolors[vertex] == gcolor;
    if (!(border[vertex] && is_simple[vertex] && is_colored)){
        return; // try again another day!
    }

    // -- read labels of neighbors --
    constexpr int MAX_NEIGHS = 32;
    uint32_t neighs_labels[MAX_NEIGHS] = {0};
    
    // -- compare all uniq points --
    uint32_t spix_id = spix[vertex];
    float3 v_ftrs = ftrs[vertex];
    float3 v_pos = pos[vertex];

    // -- selection --
    uint32_t spix_sel = spix_id;
    float curr_lprob = -1000000000;
    uint32_t curr_spix = spix_id;

    // -- iterate over neighbors --
    uint32_t start = csr_ptr[vertex];
    uint32_t end = csr_ptr[vertex+1];

    // -- read neighbor labels --
    int num_neighs = 0;
    for(uint32_t index=start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        uint32_t spix_neigh = spix[neigh];
        neighs_labels[num_neighs] = spix_neigh;
        num_neighs++;
    }


    // -- info [remove me soon] --
    // if ((end - start) > 3){
    //     printf("vertex[%d]: # of edges %d %d\n",vertex,start,end);
    // }
    // assert( (end - start) <= 3);
    //uint8_t num_neq_v = neigh_neq[vertex];
    int num = 0;
    for(uint32_t index=start; index < end; index++){
        uint32_t neigh = csr_edges[index];
        uint32_t spix_neigh = neighs_labels[num];

        uint32_t num_neq = 0;
        for(int k=0; k<num_neighs; k++){
            if (k == num) continue;
            num_neq += spix_neigh != neighs_labels[k];
        }

        // uint32_t spix_neigh = spix[neigh];
        //uint8_t num_neq = neigh_neq[neigh];
        //if (curr_spix == spix_neigh){ continue; }
        
        // float neq = num_neq + (spix_neigh != spix_id); // # different neighbors with different labels if my label switches to neighbor's label
        float neigh_lprob = compute_lprob(v_ftrs,v_pos,num_neq,params,spix_neigh,sigma2_app,potts_coeff);
        if (neigh_lprob > curr_lprob){
            curr_lprob = neigh_lprob;
            curr_spix = spix_neigh;
        }

        num++;
    }

    spix[vertex] = curr_spix;

}



__device__ float compute_lprob(float3& ftrs, float3& pos, float neigh_neq,
                               spix_params* sp_params, uint32_t sp_index,
			                   float sigma2_app, float beta){

    // -- init res --
    float lprob = -1000; // some large negative number; room for potts i think...

    // -- read --
    float3 mu_app = sp_params[sp_index].mu_app;
    double3 mu_pos = sp_params[sp_index].mu_pos;
    double3 ivar_pos = sp_params[sp_index].var_pos;
    double3 icov_pos = sp_params[sp_index].cov_pos;
    double logdet = sp_params[sp_index].logdet_sigma_shape;

    // -- appearance --
    float d_mu = (mu_app.x - ftrs.x) * (mu_app.x - ftrs.x) \
                + (mu_app.y - ftrs.y) * (mu_app.y - ftrs.y) \                 
                + (mu_app.z - ftrs.z) * (mu_app.z - ftrs.z);

    float d_pos = \
        // Diagonal terms: x_i * ivar_i * x_i
        pos.x * pos.x * ivar_pos.x +
        pos.y * pos.y * ivar_pos.y +
        pos.z * pos.z * ivar_pos.z +
        // Off-diagonal terms: 2 * x_i * icov_ij * x_j
        2.0f * pos.x * pos.y * icov_pos.x +  // 2 * x * y * icov_xy
        2.0f * pos.x * pos.z * icov_pos.y +  // 2 * x * z * icov_xz  
        2.0f * pos.y * pos.z * icov_pos.z;   // 2 * y * z * icov_yz

    // -- combine --
    lprob = - d_mu / sigma2_app - d_pos - logdet;
    lprob = lprob - beta*neigh_neq;
    //res = res - 100*(count<=10);

    return lprob;
}

