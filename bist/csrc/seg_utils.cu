

// -- basic --
#include <assert.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- project imports --
#include "seg_utils.h"
#include "init_utils.h"


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512


thrust::device_vector<int> get_unique(int* spix, int size){ // size usually "npix"

  // -- init --
  thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(spix);
  thrust::device_vector<int> uniq_spix(size);
  thrust::copy(_spix_ptr, _spix_ptr + size,uniq_spix.begin());


  // -- get unique --
  thrust::sort(uniq_spix.begin(),uniq_spix.end());
  auto uniq_end = thrust::unique(uniq_spix.begin(),uniq_spix.end());
  uniq_spix.erase(uniq_end, uniq_spix.end());
  uniq_spix.resize(uniq_end - uniq_spix.begin());
  return uniq_spix;
}

int count_invalid(int* _spix,int size){
  thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
  thrust::device_vector<int> spix(_spix_ptr,_spix_ptr+size);
  int ninvalid = thrust::count(spix.begin(), spix.end(), -1);
  return ninvalid;
}


/**********************************************
***********************************************


            Computed Sizes of Spix


***********************************************
**********************************************/

__global__
void get_spix_sizes_kernel(int* spix, int* sizes, int npix, int nspix){

  // -- get pixel index --
  int pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_idx>=npix) return;
  int batch_idx = blockIdx.y;
  pix_idx = pix_idx + npix*batch_idx;
  
  // -- superpixel at sources --
  int spix_label = spix[pix_idx];
  if (spix_label < 0){ return; } // skip invalid
  assert(spix_label < nspix);
  int* spix_size = sizes + nspix*batch_idx+spix_label;
  int size = atomicAdd(spix_size,1);
  // printf("%d\n",size);

}

__global__
void read_prior_counts_kernel(float* prior_counts,
                              spix_params* params, int nspix_buffer){
  
  // -- get superpixel index --
  int spix_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_idx>=nspix_buffer) return;

  // -- read --
  prior_counts[spix_idx] = params[spix_idx].prior_count;
}

thrust::device_vector<float> get_prior_counts(spix_params* params, int nspix_buffer){

  // -- init --
  thrust::device_vector<float> prior_counts(nspix_buffer);
  float* _prior_counts = thrust::raw_pointer_cast(prior_counts.data());

  // -- kernel --
  int nblocks = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) ); 
  dim3 Blocks(nblocks);
  dim3 NumThreads(THREADS_PER_BLOCK);
  read_prior_counts_kernel<<<Blocks,NumThreads>>>(_prior_counts, params, nspix_buffer);

  return prior_counts;
  
}


thrust::device_vector<int> get_spix_counts(int* spix, int nbatch, int npix, int nspix){

  // -- get counts --
  int* _counts = get_spix_sizes( spix, nbatch, npix, nspix);

  // -- copy to thrust --
  thrust::device_ptr<int> counts_ptr = thrust::device_pointer_cast(_counts);
  thrust::device_vector<int> counts(nspix);
  thrust::copy(counts_ptr, counts_ptr + nspix, counts.begin());
  // thrust::copy(counts.begin(),counts.end(),counts_ptr);
  cudaFree(_counts);
  return counts;
}

int* get_spix_sizes(int* spix, int nbatch, int npix, int nspix){

    // -- allocate memory --
    int* sizes = (int*)easy_allocate(nbatch*nspix,sizeof(int));
    cudaMemset(sizes, 0, nbatch*nspix*sizeof(int));

    // -- init launch info --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- shift --
    get_spix_sizes_kernel<<<BlocksPixels,NumThreads>>>(spix,sizes,npix,nspix);

    return sizes;
}


/**********************************************
***********************************************


            Find Border Pixels


***********************************************
**********************************************/


__host__ void set_border(int* seg, bool* border, int height, int width){
  int npix = height*width;
  int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  cudaMemset(border, 0, npix*sizeof(bool));
  find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,width,height);
}

__host__ void CudaFindBorderPixels(const int* seg, bool* border, const int npix,
                                   const int nbatch,const int xdim,const int ydim){
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,xdim,ydim);
}

__global__  void find_border_pixels(const int* seg, bool* border, const int npix,
                                    const int xdim, const int ydim){   
    int pix_idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (pix_idx>=npix) return; 
    int idx = pix_idx + npix*blockIdx.y; // offset via batch

    border[idx]=0;  // init        
    int x = pix_idx % xdim;
    int y = pix_idx / xdim;

    int C =  __ldg(&seg[idx]); // center 
    int N,S,E,W; // north, south, east,west            

    // -- check out of bounds --
    if ((y<1)||(x<1)||(y>=(ydim-1))||(x>=(xdim-1)))
    {
        border[idx] = 1;
        return;
    }
    N = __ldg(&seg[idx-xdim]); // above
    W = __ldg(&seg[idx-1]);  // left
    S = __ldg(&seg[idx+xdim]); // below
    E = __ldg(&seg[idx+1]);  // right
           
    // bool check0 = (C == N) and (N == W) and (W == S);
    if ( (C!=N) || (C!=S) || (C!=E) || (C!=W) ){
            border[idx]=1;  
    }


    return;        
}


/**********************************************
***********************************************


         Find Border Pixels @ END


***********************************************
**********************************************/


__host__ void CudaFindBorderPixels_end(const int* seg, bool* border, const int npix,
                                       const int nbatch,const int xdim,const int ydim){   
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    find_border_pixels_end<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,xdim,ydim);
}


__global__  void find_border_pixels_end(const int* seg, bool* border, const int npix,
                                        const int xdim, const int ydim){   
    int pix_idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (pix_idx>=npix) return; 
    int idx = pix_idx + npix*blockIdx.y; // offset via batch

    border[idx]=0;  // init        
    
    // todo; add batch info here
    int x = pix_idx % xdim;
    int y = pix_idx / xdim;

    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west            
    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init 
    
    if (y>0){
        N = seg[idx-xdim]; // above
    }          
    if (x>0){
        W = seg[idx-1];  // left
    }
    if (y<ydim-1){
        S = seg[idx+xdim]; // below
    }   
    if (x<xdim-1){
        E = seg[idx+1];  // right
    }       
   
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    // if ( (N>=0 && C!=N) || (S>=0 && C!=S) || (E>=0 && C!=E) || (W>=0 && C!=W) ){
    //         if (N>=0 && C>N) border[idx]=1; 
    //         if (S>=0 && C>S) border[idx]=1;
    //         if (E>=0 && C>E) border[idx]=1;
    //         if (W>=0 && C>W) border[idx]=1;
    // }
    // if ( ((C!=N) || (C!=S) || (C!=E) || (C!=W)) && (C>=0) ){
    if ( C>=0 ){
      if ((C>N) and (y>0)) border[idx]=1; 
      if ((C>S) and (y<ydim-1)) border[idx]=1;
      if ((C>E) and (x<xdim-1)) border[idx]=1;
      if ((C>W) and (x>0)) border[idx]=1;
    }

    return;        
}


void view_invalid(spix_params* params, int nspix){
  int nblocks = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
  dim3 Blocks(nblocks);
  dim3 NumThreads(THREADS_PER_BLOCK,1);
  view_invalid_kernel<<<Blocks,NumThreads>>>(params, nspix);
}  

__global__ void view_invalid_kernel(spix_params* params, int nspix){
  // -- get reference and query superpixel indices --
  int spix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_id>=nspix) return;
  float3 mu_app = params[spix_id].mu_app;
  int count = params[spix_id].count;
  if (count == 0){ return; }
  // -- view --
  if ((abs(mu_app.x)>10) || (abs(mu_app.y)>10) || (abs(mu_app.z)>10)){
    printf("[%d] %2.3f, %2.3f, %2.3f\n",spix_id,mu_app.x,mu_app.y,mu_app.z);
    // assert(false);
  }

}













