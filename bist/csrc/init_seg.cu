
#include "init_seg.h"
#include "init_utils.h"
#include <math.h>
#define THREADS_PER_BLOCK 512

/*************************************************

              Initialize Superpixels

**************************************************/

__host__ int init_seg(int* seg, int sp_size, int width, int height, int nbatch){

  // -- superpixel info --
  // -- sp_size is the square-root of the hexagon's area --
  int npix = height * width;
  double H = sqrt( double(pow(sp_size, 2)) / (1.5 *sqrt(3.0)) );
  double w = sqrt(3.0) * H;
  int max_num_sp_x = (int) floor(double(width)/w) + 2; // an extra "1" for edges
  int max_num_sp_y = (int) floor(double(height)/(1.5*H)) + 2; // an extra "1" for edges
  int nspix = max_num_sp_x * max_num_sp_y; //Roy -Change

  // -- launch params --
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  int nblocks_spix =  ceil(double(nspix) /double(THREADS_PER_BLOCK));
  dim3 BlockPerGrid_spix(nblocks_spix);
  int nblocks_pix =  ceil(double(npix) /double(THREADS_PER_BLOCK));
  dim3 BlockPerGrid_pix(nblocks_pix,nbatch);
  double* centers;
  cudaMalloc((void**) &centers, 2*nspix*sizeof(double));
  InitHexCenter<<<BlockPerGrid_spix,ThreadPerBlock>>>(centers, H, w, nspix,
                                                      max_num_sp_x, width, height); 
  InitHexSeg<<<BlockPerGrid_pix,ThreadPerBlock>>>(seg, centers,
                                                  nspix, npix, width);
  cudaFree(centers);
  return nspix;

}

__host__ int nspix_from_spsize(int sp_size, int width, int height){
  double H = sqrt( double(pow(sp_size, 2)) / (1.5 *sqrt(3.0)) );
  double w = sqrt(3.0) * H;
  int max_num_sp_x = (int) floor(double(width)/w) + 2;
  int max_num_sp_y = (int) floor(double(height)/(1.5*H)) + 2;
  int nspix = max_num_sp_x * max_num_sp_y;
  return nspix;
}

__global__ void InitHexCenter(double* centers, double H, double w, int nspix,
                              int max_num_sp_x, int xdim, int ydim){
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  int npix = xdim * ydim;
	if (idx >= nspix) return;
    int x = idx % max_num_sp_x; 
    int y = idx / max_num_sp_x; 
    double xx = double(x) * w;
    double yy = double(y) * 1.5 *H; 
    if (y%2 == 0){
        xx = xx + 0.5*w;
    }
    centers[2*idx]  = xx;
    centers[2*idx+1]  = yy;    
}

__global__ void InitHexSeg(int* seg, double* centers,
                           int K, int npix, int xdim){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = blockIdx.y; 	
	if (idx >= npix) return;
    int x = idx % xdim;
    int y = idx / xdim;   
    double dx,dy,d2;
    double D2 = DBL_MAX; 
    for (int j=0; j < K;  j++){
        dx = (x - centers[j*2+0]);
        dy = (y - centers[j*2+1]);
        d2 = dx*dx + dy*dy;
        if ( d2 <= D2){
              D2 = d2;  
              seg[idx+bi*npix]=j;
        }           
    } 
    return;	
}


/********************************************
     Init Square segmentation (for demo!)
 ********************************************/


__host__ int init_square_seg(int* seg, int sp_size, int width, int height, int nbatch){

  // -- superpixel info --
  int npix = height * width;
  int max_num_sp_x = (int) ceil(double(width) / sp_size);  
  int max_num_sp_y = (int) ceil(double(height) / sp_size);
  int nspix = max_num_sp_x * max_num_sp_y;
  // printf("max_num_sp_x,max_num_sp_y: %d,%d\n",max_num_sp_x,max_num_sp_y);

  // -- launch params --
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  int nblocks_spix =  ceil(double(nspix) /double(THREADS_PER_BLOCK));
  dim3 BlockPerGrid_spix(nblocks_spix,nbatch);
  int nblocks_pix =  ceil(double(npix) /double(THREADS_PER_BLOCK));
  dim3 BlockPerGrid_pix(nblocks_pix,nbatch);
  // double* centers;
  // cudaMalloc((void**) &centers, 2*nspix*sizeof(double));
  // InitSquareCenter<<<BlockPerGrid_spix,ThreadPerBlock>>>(centers, H, w, nspix,
  //                                                        max_num_sp_x, width, height); 
  InitSquareSeg<<<BlockPerGrid_pix,ThreadPerBlock>>>(seg, sp_size,max_num_sp_x, npix, width);
  // cudaFree(centers);
  return nspix;

}

__global__ void InitSquareSeg(int* seg, int sp_size,
                              int max_num_sp_x, int npix, int width){
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 	
  if (idx >= npix) return;

  int x = idx % width;
  int y = idx / width;

  // Compute superpixel index directly
  int sx = x / sp_size;
  int sy = y / sp_size;
  int sp_index = sy * max_num_sp_x + sx;

  seg[idx] = sp_index;
  // seg[idx] = 0;//sp_index;
}


/*************************************************

  Initialize 3D Superpixels (BCC Voronoi Cells)

**************************************************/

__host__ uint64_t* init_seg_3d(uint64_t* spix, float3* pos, uint32_t* bids, int* ptr, float* dim_sizes, int sp_size, int nbatch, int nnodes){
  uint64_t* nspix = (uint64_t*)easy_allocate(nbatch,sizeof(uint64_t));
  dim3 nthreads(THREADS_PER_BLOCK);
  int nblocks_nodes =  ceil(double(nnodes) /double(THREADS_PER_BLOCK));
  dim3 nblocks(nblocks_nodes);
  InitVeronoiSeg<<<nblocks,nthreads>>>(spix, nspix, pos, bids, ptr, dim_sizes, sp_size, nnodes);
  return nspix;
}


__global__ void InitVeronoiSeg(uint64_t* spix, uint64_t* nspix, float3* pos, uint32_t* bids, int* ptr, float* dim_sizes, int sp_size, int nnodes_total){
	
  // -- unpack threads --
  int node_ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (node_ix >= nnodes_total){ return; }
  int bx = bids[node_ix];
  bool cell_type = false;
             
  // -- ... --
  int ptr_offset = ptr[bx];
  int nnodes = ptr[bx+1] - ptr_offset;
  // if (bx == 1){
  //   printf("bx,nnodes: %d,%d\n",bx,nnodes);
  // }
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
        printf("[(%d,%d) (%d,%d) (%d,%d)] x,y,z,S: %d %d %d %2.2f\n",xmin,xmax,ymin,ymax,zmin,zmax,xi,yi,zi,S);
        spix[node_ix] = -1;
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
