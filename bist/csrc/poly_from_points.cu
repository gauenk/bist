/******************************************************************
 * 
 *  Extract the polygon edges from a set of vertices that 
 *  form a contiguous cluster.
 * 
 ********************************************************************/


#include "poly_from_points.h"

#define THREADS_PER_BLOCK 512

std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint32_t>>
poly_from_points(PointCloudData& data, SuperpixelParams3d& params, bool* border){

    // -- launch parameters --
    int NumThreads = THREADS_PER_BLOCK;
    int vertex_nblocks = ceil( double(data.V) / double(NumThreads) ); 
    dim3 VertexBlocks(vertex_nblocks);

    // // ...
    // int nborder_sum = thrust::count(params.border.begin(), params.border.end(), true);
    // printf("params.nspix_sum, V: %d, %d\n",params.nspix_sum,data.V);

    // thrust::host_vector<uint32_t> csum_nspix = params.csum_nspix;
    // for (int ix = 0; ix < csum_nspix.size(); ix++){
    //     printf("csum_nspix[%d]: %d\n",ix,csum_nspix[ix]);
    // }

    // -- count number of vertex per nspix --
    thrust::device_vector<uint32_t> nvertex_by_spix(params.nspix_sum, 0);
    nvertices_per_spix_kernel<<<VertexBlocks,NumThreads>>>(thrust::raw_pointer_cast(nvertex_by_spix.data()), params.spix_ptr(), params.border_ptr(), params.csum_nspix_ptr(), data.bids, data.V);

    // -- accumulate --
    thrust::device_vector<uint32_t> nvertex_by_spix_csum(params.nspix_sum+1, 0);
    thrust::inclusive_scan(nvertex_by_spix.begin(), nvertex_by_spix.end(), nvertex_by_spix_csum.begin() + 1);
    
    // thrust::host_vector<uint32_t> tmp = nvertex_by_spix_csum;
    // for(int ix = 0; ix < tmp.size(); ix++){
    //     printf("tmp[%d] = %d\n",ix,tmp[ix]);
    // }

    // exit(1);

    // -- write vertices for each --
    uint32_t V_edges = nvertex_by_spix_csum.back();
    thrust::device_vector<uint32_t> access_offsets = nvertex_by_spix_csum;
    thrust::device_vector<uint32_t> list_of_vertices(V_edges, 0);
    int vertex_edges_nblocks = ceil( double(V_edges) / double(NumThreads) ); 
    dim3 VertexEdgesBlocks(vertex_edges_nblocks);
    list_vertices_by_spix<<<VertexEdgesBlocks,NumThreads>>>(thrust::raw_pointer_cast(list_of_vertices.data()), params.spix_ptr(), params.border_ptr(), data.bids, 
                                                           params.csum_nspix_ptr(),thrust::raw_pointer_cast(access_offsets.data()),thrust::raw_pointer_cast(nvertex_by_spix_csum.data()),data.V);
        
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // exit(1);

    // -- re-order each superpixel --
    int spix_nblocks = ceil( double(params.nspix_sum) / double(NumThreads) ); 
    dim3 SpixBlocks(spix_nblocks);
    thrust::device_vector<uint32_t> ordered_verts(V_edges, 0);
    reorder_vertices_backtracking_cycle<<<SpixBlocks,NumThreads>>>(thrust::raw_pointer_cast(list_of_vertices.data()), 
                                                          data.csr_edges, data.csr_eptr, data.bids, 
                                                          params.csum_nspix_ptr(),thrust::raw_pointer_cast(nvertex_by_spix_csum.data()),
                                                          thrust::raw_pointer_cast(ordered_verts.data()),data.pos,data.V,params.nspix_sum);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
                                                 
    // exit(1);
    return {ordered_verts,nvertex_by_spix_csum};
}
 

__global__ void
nvertices_per_spix_kernel(uint32_t* nvertex_by_spix, uint32_t* spix, bool* border, uint32_t* csum_nspix, uint8_t* bids, uint32_t V){

	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];
    uint32_t spix_id = spix[vertex];
    if (!border[vertex]){ return; } // only mark border
    atomicAdd(&nvertex_by_spix[spix_id+spix_id_offset],1);
}

__global__ void
list_vertices_by_spix(uint32_t* list_of_vertices, uint32_t* spix, bool* border, uint8_t* bids, 
                      uint32_t* csum_nspix, uint32_t* access_offsets, uint32_t* nvertex_by_spix_csum, uint32_t V){

	uint32_t vertex = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (vertex>=V) return;
    uint32_t bx = bids[vertex];
    uint32_t spix_id_offset = csum_nspix[bx];


    uint32_t spix_id = spix[vertex];
    //uint32_t base_offset = nvertex_by_spix_csum[spix_id+spix_id_offset];

    uint32_t access_offset = atomicAdd(&access_offsets[spix_id+spix_id_offset],1);
    list_of_vertices[access_offset] = vertex;

}



__global__ void
reorder_vertices_backtracking_cycle(uint32_t* list_of_vertices, uint32_t* csr_edges, uint32_t* csr_eptr,
                                   uint8_t* bids, uint32_t* csum_nspix, uint32_t* nvertex_by_spix_csum,
                                   uint32_t* ordered_verts, float3* vertex_pos, uint32_t V, uint32_t S){
    
    uint32_t spix_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (spix_id >= S) return;
    
    uint32_t vertex_start_idx = nvertex_by_spix_csum[spix_id];
    uint32_t vertex_end_idx = nvertex_by_spix_csum[spix_id + 1];
    uint32_t nspix_size = vertex_end_idx - vertex_start_idx;
    
    if (nspix_size == 0) return;
    
    uint32_t* output = &ordered_verts[vertex_start_idx];
    
    for (int i = 0; i < nspix_size; i++) {
        output[i] = list_of_vertices[vertex_start_idx + i];
    }
    // return;

    // // Handle trivial cases
    // if (nspix_size <= 2) {
    //     for (int i = 0; i < nspix_size; i++) {
    //         output[i] = list_of_vertices[vertex_start_idx + i];
    //     }
    //     return;
    // }
    
    // // Safety check for array bounds
    // if (nspix_size > 256) return;
    
    // bool visited[256];
    // // uint8_t stack[256];
    // // uint8_t neigh_stack[256];
    
    // // Initialize visited array
    // for (int i = 0; i < nspix_size; i++) {
    //     visited[i] = false;
    //     // neigh_stack[i] = 0;
    // }
    
    // // Start with any boundary vertex
    // int current_idx = 0;
    // uint32_t current_vertex = list_of_vertices[vertex_start_idx];
    // output[0] = current_vertex;
    // visited[0] = true;
    
    // uint32_t prev_vertex = UINT32_MAX;
    
    // for (int pos = 1; pos < nspix_size; pos++) {
    //     uint32_t edge_start = csr_eptr[current_vertex];
    //     uint32_t edge_end = csr_eptr[current_vertex + 1];
        

    //     float3 curr_pos = vertex_pos[current_vertex];
    //     float curr_x = curr_pos.x;
    //     float curr_y = curr_pos.y;
    //     float curr_z = curr_pos.z;

    //     int best_next_idx = -1;
    //     float best_score = -1e9;
        
    //     // Evaluate all boundary neighbors
    //     for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
    //         uint32_t neighbor = csr_edges[edge_idx];
            
    //         // Skip the vertex we came from
    //         if (neighbor == prev_vertex) continue;
            
    //         // Check if neighbor is in boundary list and unvisited
    //         int neighbor_idx = -1;
    //         for (int i = 0; i < nspix_size; i++) {
    //             if (list_of_vertices[vertex_start_idx + i] == neighbor && !visited[i]) {
    //                 neighbor_idx = i;
    //                 break;
    //             }
    //         }
    //         if (neighbor_idx == -1) continue;
            
    //         float3 neigh_pos = vertex_pos[neighbor];
    //         float neigh_x = neigh_pos.x;
    //         float neigh_y = neigh_pos.y;
    //         float neigh_z = neigh_pos.z;

    //         // GEOMETRIC HEURISTIC: Choose based on angle consistency
    //         float score = 0;
            
    //         if (pos > 1) {
    //             // We have a previous direction - prefer consistent turning
    //             uint32_t prev_vertex_actual = output[pos - 2];

    //             float3 prev_pos = vertex_pos[prev_vertex_actual];
    //             float prev_x = prev_pos.x;
    //             float prev_y = prev_pos.y;
    //             float prev_z = prev_pos.z;
                
    //             // Previous direction vector
    //             float prev_dx = curr_x - prev_x;
    //             float prev_dy = curr_y - prev_y;
    //             float prev_dz = curr_z - prev_z;
                
    //             // Next direction vector  
    //             float next_dx = neigh_x - curr_x;
    //             float next_dy = neigh_y - curr_y;
    //             float next_dz = neigh_z - curr_z;
                
    //             // 3D Cross product for turn direction (gives a vector, not scalar)
    //             float cross_x = prev_dy * next_dz - prev_dz * next_dy;
    //             float cross_y = prev_dz * next_dx - prev_dx * next_dz;
    //             float cross_z = prev_dx * next_dy - prev_dy * next_dx;
    //             float cross_mag = sqrtf(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);

    //             // Dot product for angle magnitude
    //             float dot = prev_dx * next_dx + prev_dy * next_dy + prev_dz * next_dz;
    //             float prev_mag = sqrtf(prev_dx*prev_dx + prev_dy*prev_dy + prev_dz*prev_dz);
    //             float next_mag = sqrtf(next_dx*next_dx + next_dy*next_dy + next_dz*next_dz);
    //             if (prev_mag > 0 && next_mag > 0) {
    //                 float cos_angle = dot / (prev_mag * next_mag);
                    
    //                 // In 3D, we need a reference normal to define "counter-clockwise"
    //                 // Option 1: Use cross product magnitude to prefer smaller turns
    //                 float sin_angle = cross_mag / (prev_mag * next_mag);
                    
    //                 // Prefer smaller turn angles (straighter paths)
    //                 score = cos_angle - 0.5f * sin_angle; // Bias toward straight continuation
                    
    //             }
    //         } else {
    //             // First step - in 3D, establish initial direction
    //             float dx = neigh_x - curr_x;
    //             float dy = neigh_y - curr_y;
    //             float dz = neigh_z - curr_z;
                
    //             // Prefer direction with largest component (most "dominant" direction)
    //             float abs_dx = fabsf(dx);
    //             float abs_dy = fabsf(dy);
    //             float abs_dz = fabsf(dz);
    //             score = fmaxf(abs_dx, fmaxf(abs_dy, abs_dz)); // Prefer "primary" directions
                
    //         }
            
    //         if (score > best_score) {
    //             best_score = score;
    //             best_next_idx = neighbor_idx;
    //         }
    //     }
        
    //     if (best_next_idx >= 0) {
    //         uint32_t next_vertex = list_of_vertices[vertex_start_idx + best_next_idx];
    //         output[pos] = next_vertex;
    //         visited[best_next_idx] = true;
            
    //         prev_vertex = current_vertex;
    //         current_vertex = next_vertex;
    //         current_idx = best_next_idx;
    //     } else {
    //         // No valid neighbor found - should not happen in proper boundary
    //         //printf("[%d] No valid neighbor at position %d\n", spix_id, pos);
    //         break;
    //     }
    // }

}
    // // Start with first vertex
    // uint32_t start_vertex = list_of_vertices[vertex_start_idx];
    // output[0] = start_vertex;
    // visited[0] = true;
    // stack[0] = 0;
    // int stack_top = 0;  // Points to the current top element (0-based indexing)
    // int depth = 1;
    
    // int steps = 0;
    // // Main backtracking loop
    // while (stack_top >= 0 && stack_top < 256) {  // Safety bound check
    //     steps++;
    //     if (steps > 100000){
    //         printf("stack_top, depth, stack[stack_top]: %d %d %d\n",stack_top,depth,stack[stack_top]);
    //     }
    //     // if (spix_id == 0){
    //     //     printf("stack_top, depth, stack[stack_top]: %d %d %d\n",stack_top,depth,stack[stack_top]);
    //     // }
    //     // Check if we've visited all vertices
    //     if (depth == nspix_size) {
    //         // Try to complete the cycle by connecting back to start
    //         int current_idx = stack[stack_top];
    //         uint32_t current_vertex = list_of_vertices[vertex_start_idx + current_idx];
            
    //         uint32_t edge_start = csr_eptr[current_vertex];
    //         uint32_t edge_end = csr_eptr[current_vertex + 1];
            
    //         bool cycle_complete = false;
    //         for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
    //             if (csr_edges[edge_idx] == start_vertex) {
    //                 cycle_complete = true;
    //                 break;
    //             }
    //         }
            
    //         if (cycle_complete) {
    //             // SUCCESS! Perfect cycle found, output is already filled
    //             return;
    //         }
            
    //         // Can't complete cycle, backtrack
    //         visited[stack[stack_top]] = false;
    //         depth--;
    //         stack_top--;
    //         continue;
    //     }
        
    //     // Try to extend current path
    //     int current_idx = stack[stack_top];
    //     uint32_t current_vertex = list_of_vertices[vertex_start_idx + current_idx];
        
    //     uint32_t edge_start = csr_eptr[current_vertex];
    //     uint32_t edge_end = csr_eptr[current_vertex + 1];
        
    //     bool found_next = false;
    //     uint32_t start_search = edge_start + neigh_stack[stack_top];
        
    //     // Search for next unvisited boundary vertex
    //     for (uint32_t edge_idx = start_search; edge_idx < edge_end; edge_idx++) {
    //         uint32_t neighbor = csr_edges[edge_idx];
    //         neigh_stack[stack_top] = edge_idx - edge_start + 1;

    //         // Check if neighbor is in our boundary vertex list and unvisited
    //         for (int i = 0; i < nspix_size; i++) {
    //             if (list_of_vertices[vertex_start_idx + i] == neighbor && !visited[i]) {
    //                 // Found valid next vertex
    //                 visited[i] = true;
    //                 output[depth] = neighbor;
    //                 stack_top++;
    //                 stack[stack_top] = i;
    //                 neigh_stack[stack_top] = 0;  // Reset neighbor counter for new vertex
    //                 depth++;
    //                 found_next = true;
    //                 break;
    //             }
    //         }
    //         if (found_next) break;
    //     }
        
    //     // Backtrack if no valid neighbor found
    //     if (!found_next) {
    //         visited[stack[stack_top]] = false;
    //         depth--;
    //         stack_top--;
    //     }
    // }
    
    // // If we reach here, no perfect cycle was found
    // // Fall back to a simple traversal approach
    // // printf("[%d] missed the perfect cycle!\n",spix_id);
    // return;



    // //printf("[%d] missed the perfect cycle!\n",spix_id);

    // // Reset for fallback approach
    // for (int i = 0; i < nspix_size; i++) {
    //     visited[i] = false;
    // }
    
    // // Start over with greedy nearest-neighbor approach
    // output[0] = start_vertex;
    // visited[0] = true;
    
    // for (int pos = 1; pos < nspix_size; pos++) {
    //     uint32_t current_vertex = output[pos - 1];
    //     uint32_t edge_start = csr_eptr[current_vertex];
    //     uint32_t edge_end = csr_eptr[current_vertex + 1];
        
    //     bool found = false;
    //     // First try: find connected unvisited boundary vertex
    //     for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
    //         uint32_t neighbor = csr_edges[edge_idx];
    //         for (int i = 0; i < nspix_size; i++) {
    //             if (list_of_vertices[vertex_start_idx + i] == neighbor && !visited[i]) {
    //                 output[pos] = neighbor;
    //                 visited[i] = true;
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if (found) break;
    //     }
        
    //     // If no connected vertex found, just pick any unvisited one
    //     if (!found) {
    //         for (int i = 0; i < nspix_size; i++) {
    //             if (!visited[i]) {
    //                 output[pos] = list_of_vertices[vertex_start_idx + i];
    //                 visited[i] = true;
    //                 break;
    //             }
    //         }
    //     }
    // }

