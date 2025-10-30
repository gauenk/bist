
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

