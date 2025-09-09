
#include "init_seg_3d.h"
#include "init_utils.h"
#include <math.h>
#define THREADS_PER_BLOCK 512


/*************************************************

  Initialize 3D Superpixels (BCC Voronoi Cells)

**************************************************/

__host__ uint32_t* init_seg_3d(uint32_t* spix, float3* pos, uint8_t* bids, int* ptr, float* dim_sizes, int sp_size, int nbatch, int nnodes){
  uint32_t* nspix = (uint32_t*)easy_allocate(nbatch,sizeof(uint32_t));
  dim3 nthreads(THREADS_PER_BLOCK);
  int nblocks_nodes =  ceil(double(nnodes) /double(THREADS_PER_BLOCK));
  dim3 nblocks(nblocks_nodes);
  InitVeronoiSeg<<<nblocks,nthreads>>>(spix, nspix, pos, bids, ptr, dim_sizes, sp_size, nnodes);
  return nspix;
}


__global__ void InitVeronoiSeg(uint32_t* spix, uint32_t* nspix, float3* pos, uint8_t* bids, int* ptr, float* dim_sizes, int sp_size, int nnodes_total){
	
  // -- unpack threads --
  int node_ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (node_ix >= nnodes_total){ return; }
  int bx = bids[node_ix];
  bool cell_type = false;
             
  // -- ... --
  int ptr_offset = ptr[bx];
  int nnodes = ptr[bx+1] - ptr_offset;
  float S = 0.01 * sp_size; // about 1 cm spacing.
  int local_node_ix = node_ix - ptr_offset;
  if (local_node_ix >= nnodes){ return; }

  // -- bounds in 3d space --
  // int xlen = dim_sizes[6*bx+1] - dim_sizes[6*bx+0];
  // int ylen = dim_sizes[6*bx+3] - dim_sizes[6*bx+2];
  // int zlen = dim_sizes[6*bx+5] - dim_sizes[6*bx+4];
  int xmin = floorf(dim_sizes[6*bx+0]/S)-1;
  int xmax = ceilf(dim_sizes[6*bx+1]/S)+1;
  int ymin = floorf(dim_sizes[6*bx+2]/S)-1;
  int ymax = ceilf(dim_sizes[6*bx+3]/S)+1;
  int zmin = floorf(dim_sizes[6*bx+4]/S)-1;
  int zmax = ceilf(dim_sizes[6*bx+5]/S)+1;


  // -- unpack position --
  float3 xyz = pos[node_ix];
  float x = xyz.x;
  float y = xyz.y;
  float z = xyz.z;
  // printf("x,y,z,S: %2.2f %2.2f %2.2f %2.2f\n",x,y,z,S);

  // -- lattic coordinates --
  int xi_a = __float2int_rn((float)x / S); // round nearest (_rn)
  int yi_a = __float2int_rn((float)y / S); 
  int zi_a = __float2int_rn((float)z / S); 
  int xi_b = __float2int_rd((float)x / S); // round down (_rd)
  int yi_b = __float2int_rd((float)y / S); 
  int zi_b = __float2int_rd((float)z / S); 

  // -- center A --
  float distA = (x - S*xi_a)*(x - S*xi_a)
                  + (y - S*yi_a)*(y - S*yi_a)
                  + (z - S*zi_a)*(z - S*zi_a);
  float distB = (x - S*xi_b + 0.5)*(x - S*xi_b + 0.5)
                  + (y - S*yi_b + 0.5)*(y - S*yi_b + 0.5)
                  + (z - S*zi_b + 0.5)*(z - S*zi_b + 0.5);

  // -- select point type --
  int xi,yi,zi;
  if (distA < distB){
    xi = xi_a;
    yi = yi_a;
    zi = zi_a;
    cell_type = false;
  }else{
    xi = xi_b;
    yi = yi_b;
    zi = zi_b;
    cell_type = true;
  }

  // -- ensure boundary --
  if ((xi < xmin) || (xmax < xi) ||
      (yi < ymin) || (ymax < yi) ||
      (zi < zmin) || (zmax < zi)){
        //printf("[(%d,%d) (%d,%d) (%d,%d)] x,y,z,S: %d %d %d %2.2f\n",xmin,xmax,ymin,ymax,zmin,zmax,xi,yi,zi,S);
        spix[node_ix] = UINT32_MAX;
        return;
      }

  // -- start numbering at (0,0,0) --
  xi -= xmin;
  yi -= ymin;
  zi -= zmin;

  // -- ... --
  // int nx = (xmax - xmin - 1)/S+1;
  // int ny = (ymax - ymin - 1)/S+1;
  // int nz = (zmax - zmin - 1)/S+1;
  int nx = xmax - xmin;
  int ny = ymax - ymin;
  int nz = zmax - zmin;

  spix[node_ix] = (xi * ny * nz + yi * nz + zi) * 2 + cell_type;
  //printf("spix[%d] = %d\n",node_ix,spix[node_ix]);
  if (local_node_ix == 0){
    nspix[bx] = min(nx *  ny * nz * 2,nnodes);
  }
}
