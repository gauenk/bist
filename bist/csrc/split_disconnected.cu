


// -- basic imports --
#include <assert.h>

// -- cuda imports --
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#define THREADS_PER_BLOCK 512

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>

// -- project imports --
#include "structs.h"
#include "init_utils.h"
#include "seg_utils.h"
#include "split_disconnected.h"
#include "atomic_helpers.h"

#define THREADS_PER_BLOCK 512

__host__ void
run_invalidate_disconnected(int* seg, int nbatch, int height, int width, int nspix){

    // -- better names --
    int H = height;
    int W = width;
    int npix = H*W;

    // -- init launch info --
    int nblocks_for_npix = ceil(double(npix)/double(THREADS_PER_BLOCK)); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    int nblocks_for_nspix = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksSuperPixels(nblocks_for_nspix,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 NumThreads(THREADS_PER_BLOCK,1);
    const int sizeofint = sizeof(int);

    // -- get regions --
    auto out0 = get_regions(seg,nbatch,H,W);
    int* regions = std::get<0>(out0);
    int* region_ids = std::get<1>(out0);
    int* region_to_spix = std::get<2>(out0);
    int  nregions = std::get<3>(out0);

    // -- [dev only] get sizes of spix --
    int* spix_sizes = get_spix_sizes(seg, nbatch, npix, nspix);

    // -- get size of each regions --
    int* region_sizes = (int*)easy_allocate(nbatch*nregions,sizeof(int));
    compute_region_sizes<<<BlocksPixels,NumThreads>>>(
              region_sizes,regions,npix,nregions);

    // -- assign each spix to its largest region --
    uint64_t* tmp_spix_to_regions=(uint64_t*)easy_allocate(nbatch*nspix,sizeof(uint64_t));
    cudaMemset(tmp_spix_to_regions, 0, nbatch*nspix*sizeof(uint64_t));
    int nblocks_for_nregions = ceil( double(nregions) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksRegions(nblocks_for_nregions,nbatch);
    assign_spix_to_regions<<<BlocksRegions,NumThreads>>>(\
       tmp_spix_to_regions,region_to_spix,region_sizes,nregions,nspix);

    // -- decoded each region and mark valid regions --
    int* spix_to_regions = (int*)easy_allocate(nbatch*nspix,sizeof(int));
    bool* valid_regions =  (bool*)easy_allocate(nbatch*nregions,sizeof(bool));
    cudaMemset(spix_to_regions, -1, nbatch*nspix*sizeof(int));
    cudaMemset(valid_regions, false, nbatch*nregions*sizeof(bool));
    decode_spix_to_regions<<<BlocksSuperPixels,NumThreads>>>(\
        spix_to_regions,spix_sizes,valid_regions,
        tmp_spix_to_regions,nregions,nspix);

    // -- invalidate disconnected superpixels --
    invalidate_disconnected_spix<<<BlocksPixels,NumThreads>>>(\
       seg,regions,valid_regions,npix);

    // -- free --
    cudaFree(valid_regions);
    cudaFree(spix_to_regions);
    cudaFree(tmp_spix_to_regions);
    cudaFree(spix_sizes); // dev only
    cudaFree(region_sizes);
    cudaFree(region_to_spix);
    cudaFree(region_ids);
    cudaFree(regions);

    return;
}


/**************************************************

    Get each connected region
    AND its corresponding supeprixel

***************************************************/


__host__
std::tuple<int*,int*,int*,int>
get_regions(int* seg, int nbatch, int H, int W){

    // -- allocate --
    int npix = H*W;
    thrust::device_vector<int> regions(nbatch*npix);
    thrust::sequence(regions.begin(), regions.end(), 0);
    thrust::device_vector<int> prev(regions.begin(), regions.end());
    

    // -- pointers --
    int* regions_ptr = thrust::raw_pointer_cast(regions.data());
    int* prev_ptr = thrust::raw_pointer_cast(prev.data());
    const int sizeofint = sizeof(int);

    // -- blocks --
    int nblocks_for_npix = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlocksPixels(nblocks_for_npix,nbatch);
    dim3 NumThreads(THREADS_PER_BLOCK,1);

    // -- allocate [change] --
    int changes = 1;
    int* changes_gpu = (int*)easy_allocate(1,sizeof(int));

    // -- [dev] START timing --
    // clock_t start,finish;
    // cudaDeviceSynchronize();
    // start = clock();

    // -- compute minimum pixel index within each connected region --
    // int _dev_count = 0;
    while(changes){
      cudaMemcpy(prev_ptr,regions_ptr,npix*sizeofint,cudaMemcpyDeviceToDevice);
      cudaMemset(changes_gpu, 0, sizeofint);
      cudaMemcpy(&changes, changes_gpu, sizeofint, cudaMemcpyDeviceToHost);
      find_spix_min<<<BlocksPixels,NumThreads>>>(seg,regions_ptr,prev_ptr,
                                                 changes_gpu,H,W,npix);
      cudaMemcpy(&changes,changes_gpu,sizeofint,cudaMemcpyDeviceToHost);
      // _dev_count++;
      // if (_dev_count > 100){
      //   std::cout << changes << std::endl;
      // }
    }
    cudaFree(changes_gpu); // -- free --

    // -- [dev] FINISH timing --
    // cudaDeviceSynchronize();
    // finish = clock();
    // std::cout << "[this] Chunk Takes " << \
    //   ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << std::endl;

    // -- uniq regions --
    thrust::device_vector<int> ids(regions.begin(), regions.end());
    thrust::sort(ids.begin(),ids.end());
    auto uniq_end = thrust::unique(ids.begin(),ids.end());
    ids.resize(uniq_end-ids.begin());
    int* ids_ptr = thrust::raw_pointer_cast(ids.data());
    int nregions = ids.size();

    // -- relabel regions --
    relabel_spix<<<BlocksPixels,NumThreads>>>(regions_ptr,ids_ptr,npix,nregions);

    // -- find the spix for each region --
    thrust::device_vector<int> spix_tr(seg,seg+nbatch*npix);
    thrust::device_vector<int> region_spix(nbatch*nregions);
    thrust::gather(ids.begin(),ids.end(),spix_tr.begin(),region_spix.begin());
    int* region_spix_ptr = thrust::raw_pointer_cast(region_spix.data());

    // -- copy data so thrust doesn't destroy memory --
    int* regions_rtn = (int*)easy_allocate(nbatch*npix,sizeof(int));
    cudaMemcpy(regions_rtn,regions_ptr,nbatch*npix*sizeof(int),cudaMemcpyDeviceToDevice);
    int* ids_rtn = (int*)easy_allocate(nbatch*nregions,sizeof(int));
    cudaMemcpy(ids_rtn,ids_ptr,nbatch*nregions*sizeof(int),cudaMemcpyDeviceToDevice);
    int* region_spix_rtn = (int*)easy_allocate(nbatch*nregions,sizeof(int));
    cudaMemcpy(region_spix_rtn,region_spix_ptr,nbatch*nregions*sizeof(int),cudaMemcpyDeviceToDevice);

    return std::make_tuple(regions_rtn,ids_rtn,region_spix_rtn,nregions);
}

// __host__
// std::tuple<torch::Tensor,int>
// get_split_starts(torch::Tensor spix_counts){

//     // -- get the number of superpixel occurances per region --
//     spix_counts = spix_counts - 1; // size()[0] = nspix
//     auto args = torch::where(spix_counts <= 0).at(0);
//     auto zero = torch::tensor({0}, torch::kInt32);
//     spix_counts = spix_counts.index_put_({args}, zero);
//     int num_splits = torch::sum(spix_counts.index({args})).item<int>();

//     // -- put the starting additional spix num at the correct location --
//     // auto split_starts = -torch::ones_like(spix_counts);
//     auto split_starts = torch::zeros_like(spix_counts);
//     args = torch::where(spix_counts > 0).at(0);
//     auto cumsum_elements = torch::cumsum(spix_counts.index({args}),0).to(torch::kInt32);
//     spix_counts = spix_counts.index_put_({args}, cumsum_elements);

//     // base case; no split spix
//     if (args.sizes()[0] == 0){
//       return std::make_tuple(split_starts,0);
//     }

//     // -- take max number of splits for a single superpixel --
//     int max_nsplits = torch::max(cumsum_elements).item<int>();

//     // -- finally, get the starting index for each split --
//     // args = th.where(spix_conts>0)
//     // split_starts = -th.ones_like(spix_counts)
//     // split_starts[args] = th.cumsum(spix_counts[args]) - spix_counts[args[0]]
//     auto first_num = spix_counts.index({args[0]});
//     assert(first_num.item<int>()>=1);//must be at least one.
//     // fprintf(stdout,"first num: %d\n",first_num.item<int>());
//     split_starts = split_starts.index_put_({args}, cumsum_elements) - first_num;

//     return std::make_tuple(split_starts,max_nsplits);
// }


/**************************************************


               Kernels Start Here


***************************************************/


/**************************************************

   1. Assign one region (the largest) to each spix
      -> this requires coding to ensure the opt is atomic
   2. Decoded the coded regions, and mark the valid regions
   3. Invalidate all the spix locates associated
      with a disconnected region

***************************************************/



__global__
void assign_spix_to_regions(uint64_t* tmp_spix_to_regions,
                            int* region_to_spix,
                            int* region_sizes, int nregions, int nspix){
  
  // -- get region id --
  int region_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (region_id>=nregions)  return;

  // -- unpack --
  int spix_id = region_to_spix[region_id];
  int region_size = region_sizes[region_id];
  assert(spix_id>=0);
  assert(spix_id<nspix);

  // -- compare and update --
  atomic_max_update(tmp_spix_to_regions+spix_id, region_size, region_id);

}

__global__
void decode_spix_to_regions(int* spix_to_regions,
                            int* spix_sizes,
                            bool* valid_regions,
                            uint64_t* tmp_spix_to_regions,
                            int nregions, int nspix){

  // -- get superpixel index --
  int spix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_id>=nspix)  return;
  int spix_size = spix_sizes[spix_id];
  if (spix_size == 0){ return; }

  // -- decode --
  uint64_t combined = tmp_spix_to_regions[spix_id];
  int decoded_region = int(uint32_t(combined));
  assert(decoded_region >= 0);
  assert(decoded_region < nregions);

  // -- [dev only] --
  // if (spix_id == 0){
  //   printf("spix_id,decoded_region: %d,%d\n",spix_id,decoded_region);
  // }

  // -- store --
  spix_to_regions[spix_id] = decoded_region;
  valid_regions[decoded_region] = true;
}

__global__
void invalidate_disconnected_spix(int* spix, int* regions,
                                  bool* valid_regions, int npix){

  // -- get pixel index --
  int pix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (pix_id>=npix)  return;

  // -- invalidate if disconnected --
  int region = regions[pix_id];
  bool valid = valid_regions[region];
  if (not(valid)){
    // if (spix[pix_id] == 0){ printf(".\n"); } // dev only
    spix[pix_id] = -1;
  }
}




/**************************************************



        ***** New Information Here ******



***************************************************/


__global__
void update_spix(int* spix, int* regions, int* children,
                 int* split_starts, int* region_offsets,
                 int npix, int nspix, int max_nchild){

  // -- get pixel index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  
  // -- get index --
  int* spix_idx = spix + idx;
  int region_idx = *(regions + idx);
  int split_start = *(split_starts + spix_idx[0]);
  int region_offset = *(region_offsets + region_idx);
  int* child_ptr = children + spix_idx[0]*max_nchild;
  // int region_offset = *(region_offsets + spix_idx[0]*(max_nchild+1));
  // int region_offset = *(region_offsets + spix_idx[0]*(max_nchild+1) + region_idx);

  // if (split_start >= 0){ // only non-negative if valid
  // // if (region_offset >= 0){
  //   assert(region_offset>=0);
  if (region_offset>0){
    assert(split_start >= 0);
    spix_idx[0] = nspix + split_start + region_offset;
    (child_ptr + region_offset-1)[0] = nspix + split_start + region_offset;
  }

}

__global__
void compute_offsets(int* regions_spix, int* region_sizes,
                     int* region_offsets, int nspix, int nregions, int max_nchild){

  // -- get spix index --
  int spix = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix>=nspix)  return;
  // ? shouldn't this be "nregions" size?

  // -- the loop --
  // int r_max = -1;
  // int r_zero = -1;
  int curr_max = -1;
  int child_count = 0;
  int idx_max;
  int idx_zero;
  for (int r=0; r < nregions; r++){
    if (regions_spix[r] == spix){

      // -- get zero index --
      if (child_count == 0){
        idx_zero = r;
      }

      // -- get offset --
      region_offsets[r] = child_count;

      // -- get max size --
      int count = region_sizes[r];
      if (count > curr_max){
        idx_max = r;
        curr_max = count;
      }

      // -- update child count --
      child_count++;

    }
  }

  // -- swap largest region to zero offset --
  if ((idx_max != idx_zero) and (child_count > 0)){
    region_offsets[idx_zero] = region_offsets[idx_max];
    region_offsets[idx_max] = 0;
    // region_offsets[spix*max_nchild] = idx_max;
    // region_offsets[spix*max_nchild+idx_max] = 0;
    // region_offsets[spix*max_nchild+r_zero] = region_offsets[spix*max_nchild+r_max];
    // region_offsets[spix*max_nchild+r_max] = 0;
  }


}

__global__
void get_spix_counts(int* regions_spix, int* spix_counts, int nregions){
  /*************************
   
     "regions_spix" is the superpixel associated with each region; length "nregions"
     "spix_counts" is the number of counts associated with each spix; length "nspix"

  *************************/
  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=nregions)  return;
  int spix = regions_spix[idx];
  atomicAdd(&spix_counts[spix],1);
}

__global__
void compute_region_sizes(int* size, int* regions, int npix, int nregions){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=npix)  return;
  int region = regions[idx];
  assert(region<nregions);
  atomicAdd(&size[region],1);
}


__global__
void relabel_spix(int* spix, int* ids, int npix, int num_ids){

  // -- filling superpixel params into image --
  // extern __shared__ int buff[];
  int ix = threadIdx.x + blockIdx.x * blockDim.x;  
  if (ix>=npix) return; 

  // -- offset super pixels --
  int spix_ix = *(spix + ix);
  int new_id = -1;

  // -- offset of kx -- [binary search not needed]
  for (int kx=0; kx<num_ids; kx++){
    if (ids[kx] == spix_ix){
      new_id = kx;
      break;
    }
  }
  (spix + ix)[0] = new_id;
}


__global__
void find_spix_min(int* seg, int* curr, int* prev,
                   int* changes, int H, int W, int npix){
  
  /****************************************

     Assign all pixels in a contiguous region to
     the same (here minimum) x,y coordinate.

     A bit more than 10% of the total runtime
     of forward propogation is spend in this single kernel...

  *****************************************/

  // -- get index --
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  // atomicAdd(&changes[0],1);
  if (idx>=npix)  return;
  int hi = idx / W;
  int wi = idx % W;

  // -- only check locally --
  int neigh_idx;
  bool oob,match;
  #pragma unroll
    for (int jx=-1;jx<=1;jx++){
#pragma unroll
    for (int ix=-1;ix<=1;ix++){
      oob = check_oob(hi,ix,wi,jx,H,W);
      // if (oob){ continue; }
      if (not oob){
        neigh_idx = (hi+ix)*W + (wi+jx);
        match = seg[idx] == seg[neigh_idx];
        if (match){  curr[idx] = min(curr[idx],prev[neigh_idx]); }
      }
    }
  }

  // -- detect changes --
  atomicAdd(changes,curr[idx] != prev[idx]);

}

// /**********************************************************

//              -=-=-=-=- Python API  -=-=-=-=-=-

// ***********************************************************/

// std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>
// split_disconnected(const torch::Tensor spix, int nspix){

//   // -- check input --
//   CHECK_INPUT(spix);

//   // -- unpack --
//   int nbatch = spix.size(0);
//   int height = spix.size(1);
//   int width = spix.size(2);
//   assert(nbatch==1); // currently, must be batchsize of 1
  
//   // -- main function --
//   auto fxn_outs = run_split_disconnected(spix.data<int>(), nbatch,
//                                          height, width, nspix);
//   // -- return --
//   // return std::make_tuple(spix,children);
//   auto full_outs = std::tuple_cat(std::make_tuple(spix),fxn_outs);
//   return full_outs;

// }


