
/********************************************************************

      Run BASS using the propograted superpixel segs and params

********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <assert.h>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// -- "external" import --
// #include "structs.h"

// -- utils --
#include "init_utils.h"
#include "init_seg.h"
#include "flood_fill.h"
#include "compact_spix_3d.h"
#include "structs_3d.h"
#include "update_params_3d.h"
#include "update_seg_3d.h"
#include "sparams_io_3d.h"
#include "seg_utils_3d.h"

#define THREADS_PER_BLOCK 512


// /**********************************************************

//              -=-=-=-=- Main Function -=-=-=-=-=-

// ***********************************************************/

__host__ void bass(spix_params* aos_params, spix_helper* sm_helper, PointCloudData& data, SuperpixelParams3d& soa_params, SpixMetaData& args, Logger* logger){

  // -- init --
  // todo; batchify both fxn; right now spix_ids are 0 - nspix for each batch so indexing is off.. easy fix just not done yet.
  for (int idx = 0; idx < args.niters; idx++) {

    // -- Update Parameters --
    update_params(aos_params,sm_helper,data,soa_params,args,logger);

    // -- Update Segmentation --
    update_seg(aos_params,sm_helper,data,soa_params,args,logger);

  }

  // -- set one-sided border at end for nicer viz --
  cudaMemset(soa_params.border_ptr(), 0, data.V*sizeof(bool));
  set_border_end(soa_params.spix_ptr(),soa_params.border_ptr(),data.csr_edges,data.csr_eptr,data.V);

}


// // Cumulative sum & max over nspix
// uint64_t* nspix_csum = (uint64_t*)easy_allocate((nbatch+1),sizeof(uint64_t));
// uint64_t nspix_max;
// {
//   cudaMemset(nspix_csum, 0, sizeof(uint64_t));
//   thrust::device_ptr<uint64_t> in_ptr(init_nspix);
//   thrust::device_ptr<uint64_t> out_ptr(nspix_csum);
//   thrust::inclusive_scan(thrust::device, in_ptr, in_ptr + nbatch, out_ptr+1);
//   uint64_t max_val = thrust::reduce(thrust::device,in_ptr, in_ptr + nbatch,(uint64_t)0,thrust::maximum<uint64_t>());
//   cudaMemcpy(&nspix_max, &max_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
// }


__global__ void init_mark_active_kernel(spix_params* sp_params, uint32_t* spix, uint8_t* vbids, uint32_t* csum_nspix, int V) {
    // -- filling superpixel params into image --
    int vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex >= V) return;
    uint8_t bx = vbids[vertex];
    uint32_t spix_offset = csum_nspix[bx];
    uint32_t spix_id = spix[vertex];
    sp_params[spix_id+spix_offset].valid = 1;
}

void mark_active_init(spix_params* sp_params, uint32_t* spix, uint8_t* vbids, uint32_t* csum_nspix, int V){
  int nthreads = 256;
  int nblocks = (V - 1) / nthreads + 1;
  init_mark_active_kernel<<<nblocks,nthreads>>>(sp_params,spix,vbids,csum_nspix,V);
  return;
}

/**********************************************************

             -=-=-=-=- C++/Python API  -=-=-=-=-=-

***********************************************************/

SuperpixelParams3d run_bass3d(PointCloudData& data, SpixMetaData& args, Logger* logger){

    // -- init spix --
    SuperpixelParams3d params(data.V,data.B);

    // -- init spix & compact  [ sets nspix within params ] --
    uint32_t* init_nspix = init_seg_3d(params.spix_ptr(), data.pos, data.bids, data.ptr, data.dim_sizes, args.sp_size, data.B, data.V);
    run_compactify(params.nspix_ptr(), params.spix_ptr(), data.bids, params.prev_nspix_ptr(), init_nspix, data.B, data.V);
    params.comp_csum_nspix();
    cudaDeviceSynchronize();
    cudaFree(init_nspix);

    // -- allocate memory --
    int nspix_sum = params.comp_nspix_sum();
    params.set_nspix_sum(nspix_sum);
    int nspix_max = params.nspix_sum*args.nspix_buffer_mult;
    printf("nspix_max: %d\n",nspix_max);
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    spix_params* sp_params=(spix_params*)easy_allocate(nspix_max,sparam_size);
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_max,helper_size);

    // -- init sp_params --
    mark_active_init(sp_params,params.spix_ptr(),data.bids,params.csum_nspix_ptr(),data.V);

    // -- ensure clusters are contiguous --
    update_params(sp_params,sp_helper,data,params,args,logger);
    flood_fill(data,sp_params,params.spix_ptr(),params.csum_nspix_ptr(),nspix_sum);

  
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //                 Run BASS 3D
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- run method --
    bass(sp_params, sp_helper, data, params, args, logger);

    // -- read superpixel info --
    aos_to_soa(sp_params, params);

//     // // -- view --
//     // thrust::device_vector<int> uniq_spix(_spix_ptr, _spix_ptr + npix);
//     // thrust::sort(uniq_spix.begin(),uniq_spix.end());
//     // auto uniq_end = thrust::unique(uniq_spix.begin(),uniq_spix.end());
//     // uniq_spix.erase(uniq_end, uniq_spix.end());
//     // uniq_spix.resize(uniq_end - uniq_spix.begin());
//     // printf("delta: %d\n",uniq_end - uniq_spix.begin());
//     // int nactive = uniq_spix.size();
//     // int* _uniq_spix = thrust::raw_pointer_cast(uniq_spix.data());
//     // printf("nactive: %d\n",nactive);
//     // int _num_blocks = ceil( double(nactive) / double(THREADS_PER_BLOCK) ); 
//     // dim3 _nblocks(_num_blocks);
//     // dim3 _nthreads(THREADS_PER_BLOCK);
//     // _view_prior_counts_kernel<<<_nblocks,_nthreads>>>(sp_params, _uniq_spix, nactive);


//     // -- extract uniq ids --
//     auto [prop_ids, _new_nspix] = extract_unique_ids_batch_cub(_spix,prev_nspix,nbatch,npix);

//   // {
//   //   // Copy device vector to host
//   //   thrust::host_vector<int> h_nspix = _new_nspix;

//   //   // Print the values
//   //   std::cout << "New nspix: "; 
//   //   std::cout << h_nspix.size() << std::endl;
//   //   for (int i = 0; i < h_nspix.size(); i++) {
//   //       std::cout << h_nspix[i] << " ";
//   //   }
//   //   std::cout << std::endl;
    
//   // }
//     // {
//     //   thrust::device_vector<int> _nspix(nspix.size());
//     //   thrust::transform(_new_nspix.begin(), _new_nspix.end(),prev_nspix.begin(),_nspix.begin(),thrust::plus<int>());
//     //   bool check_eq = thrust::equal(nspix.begin(), nspix.end(), _nspix.begin());
//     //     if (!check_eq) {
//     //     printf("ERROR: nspix don't match\n");
//     //   }
//     // }
//     //assert(nspix == new_nspix + prev_nspix);
//     compactify_new_superpixels_b(_spix,sp_params,prop_ids,_new_nspix,prev_nspix,nbatch,nspix_buffer,npix);

//     // -- get spixel parameters as tensors --
//     // thrust::device_vector<int> uniq_ids = get_unique(_spix,npix);
//     // int num_ids = uniq_ids.size();
//     // int* _uniq_ids = thrust::raw_pointer_cast(uniq_ids.data());
//     // SuperpixelParams* params = get_params_as_vectors(sp_params,_uniq_ids,num_ids,nspix);
//     // run_update_prior(params,_uniq_ids, npix, nspix, 0,false);
//     SuperpixelParams* params = nullptr;

//     // ...
//     CudaFindBorderPixels_end(_spix,border,npix,nbatch,width,height);

//     gpuErrchk( cudaPeekAtLastError() );
//     gpuErrchk( cudaDeviceSynchronize() );


    // -- free --
    cudaFree(sp_helper);
    cudaFree(sp_params);

    // -- return! --
    return params;
}


