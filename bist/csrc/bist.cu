/********************************************************************

      BIST main function [propogrates superpixel segs and params]

********************************************************************/

// -- cpp imports --
#include <stdio.h>
#include <assert.h>
#include <tuple>

// -- thrust --
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h> // for debugging

// -- "external" import --
#include "structs.h"

// -- utils --
#include "seg_utils.h"
#include "init_utils.h"
#include "init_seg.h"
#include "init_sparams.h"
#include "compact_spix.h"
#include "sparams_io.h"
#include "relabel.h"

// -- primary functions --
#include "split_merge_prop.h"
#include "update_params.h"
#include "update_seg.h"

#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

std::tuple<int,int> _get_min_max(int* _spix, int npix){
    // -- init superpixels --
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    auto min_iter = thrust::min_element(spix.begin(), spix.end());
    auto max_iter = thrust::max_element(spix.begin(), spix.end());
    int min_seg = *min_iter;
    int max_seg = *max_iter;
    return std::make_tuple(min_seg,max_seg);
}
void _print_min_max(int* _spix, int npix){
    int min_seg, max_seg;
    std::tie(min_seg, max_seg) = _get_min_max(_spix,npix);
    std::cout << "Minimum element: " << min_seg << std::endl;
    std::cout << "Maximum element: " << max_seg << std::endl;
}


__host__ int run_bist_alg(float* img, int* seg, int* shifted,
                          spix_params* sp_params,
                          bool* border,spix_helper* sp_helper,
                          spix_helper_sm_v2* sm_helper,
                          // spix_helper_sm* sm_helper,
                          int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                          int niters, int niters_seg, int sm_start,
                          float sigma2_app,  float sigma2_size, int sp_size,
                          float potts, float alpha_hastings,
                          int nspix, int nspix_buffer,
                          int nbatch, int width, int height, int nftrs,
                          SuperpixelParams* params_prev, int nspix_prev,
                          float epsilon_reid, float epsilon_new,
                          float merge_alpha, float split_alpha, float gamma,
                          int target_nspix, bool prop_flag, Logger* logger){

    // -- init --
    int count = 1;
    int npix = height * width;
    int max_spix = nspix-1;
    
    // -- controlled nspix --
    int og_niters = niters;
    bool nspix_controlled = target_nspix>0;
    if (nspix_controlled){
      std::cout << "target_nspix: " << target_nspix << std::endl;
      niters = 5000;
    }


    for (int idx = 0; idx < niters; idx++) {

      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

      // -- control split/merge to yield a fixed # of spix --
      if (nspix_controlled){
        if ((idx % 2) == 0){
          thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
          int nliving = prop_ids.size();
          if (nliving > 1.05*target_nspix){ 
            if (split_alpha > 0){
              split_alpha = 0;
              merge_alpha = 0;
            } // reset
            split_alpha += -1;
            merge_alpha += 1.0;
          }else if (nliving < 0.95*target_nspix){
            if (split_alpha < 0){
              split_alpha = 0;
              merge_alpha = 0;
            } // reset
            split_alpha += 1;
            merge_alpha += -1;
          }else{
            split_alpha = 0.;
            merge_alpha = 0.;
          }
          if ((idx > 2000) and (idx%200 == 0)){ // mark to new if  can't fix # spix
            epsilon_new = 2*(epsilon_new+1e-5);
          }
          bool accept_cond = (nliving < 1.05*target_nspix);
          accept_cond = accept_cond and (nliving > 0.95*target_nspix);
          accept_cond = accept_cond and (idx >= og_niters);
          if (accept_cond){ break; }
        }
      }

      // -- Run Split/Merge --
      if (idx > sm_start){
        if(idx%4 == 1){
          // count = 2;
          max_spix = run_split_p(img, seg, shifted, border, sp_params,
                                 sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                                 alpha_hastings, split_alpha, gamma,
                                 sigma2_app, sigma2_size,
                                 count, idx, max_spix,
                                 sp_size,npix,nbatch,width,
                                 height,nftrs,nspix_buffer,logger);

          // exit(1);
          // gpuErrchk( cudaPeekAtLastError() );
          // gpuErrchk( cudaDeviceSynchronize() );

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

        }
        if( idx%4 == 3){
          thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
          max_spix = relabel_spix(seg,sp_params,params_prev,prop_ids,
                                  epsilon_reid,epsilon_new,
                                  height,width,nspix_prev,max_spix,logger);
          run_merge_p(img, seg, border, sp_params,
                      sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                      merge_alpha,alpha_hastings, sigma2_app, sigma2_size,count,idx,
                      max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer,logger);

          // -- Update Parameters --
          update_params(img, seg, sp_params, sp_helper, sigma2_app,
                        npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

        }
      }


      // -- Update Segmentation --
      update_seg(img, seg, border, sp_params,
                 niters_seg, sigma2_app, potts,
                 npix, nbatch, width, height, nftrs, logger);
      
      // gpuErrchk( cudaPeekAtLastError() );
      // gpuErrchk( cudaDeviceSynchronize() );



    }

    update_params(img, seg, sp_params, sp_helper, sigma2_app,
                  npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);
    store_sample_sigma_shape(sp_params,sp_helper,sp_size, nspix_buffer);

    if (nspix_controlled){
      thrust::device_vector<int> prop_ids = extract_unique_ids(seg, npix, 0);
      int nliving = prop_ids.size(); // only when controlling # spix
      //printf("nliving: %d\n",nliving);
    }

    // printf("end core bist\n");
    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;

}


int get_max_spix(int* _spix, int npix){
    // -- init superpixels --
    thrust::device_ptr<int> _spix_ptr = thrust::device_pointer_cast(_spix);
    thrust::device_vector<int> spix(_spix_ptr, _spix_ptr + npix);

    // auto min_iter = thrust::min_element(spix.begin(), spix.end());
    auto max_iter = thrust::max_element(spix.begin(), spix.end());
    int max_seg = *max_iter;
    return max_seg;
}

/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

std::tuple<int*,bool*,SuperpixelParams*>
run_bist(float* img, int nbatch, int height, int width, int nftrs,
         int niters, int niters_seg, int sm_start, int sp_size,
         float sigma2_app, float sigma2_size,
         float potts, float alpha_hastings,
         int* spix_prev, int* shifted_spix, SuperpixelParams* params_prev,
         float epsilon_reid, float epsilon_new,
         float merge_alpha, float split_alpha,
         float gamma, int target_nspix, bool prop_flag, Logger* logger){


    // -- unpack --
    int npix = height*width;
    assert(nbatch==1);    
    int nspix = params_prev->ids.size();
    
    // -- allocate filled spix --
    int* _spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
    cudaMemcpy(_spix, spix_prev, npix*sizeof(int), cudaMemcpyDeviceToDevice);
    int nspix_prev = nspix;
    //printf("nspix_prev: %d\n",nspix_prev);

    // // -- allocate memory --
    int nspix_buffer = nspix*10;
    const int sparam_size = sizeof(spix_params);
    const int helper_size = sizeof(spix_helper);
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    spix_helper* sp_helper=(spix_helper*)easy_allocate(nspix_buffer,helper_size);
    int* sm_seg1 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_seg2 = (int*)easy_allocate(npix,sizeof(int));
    int* sm_pairs = (int*)easy_allocate(2*nspix_buffer,sizeof(int));
    const int sm_helper_size = sizeof(spix_helper_sm_v2);
    spix_helper_sm_v2* sm_helper=(spix_helper_sm_v2*)easy_allocate(nspix_buffer,
                                                                sm_helper_size);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //             Run BIST
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- init params --
    spix_params* sp_params = get_vectors_as_params(params_prev,sp_size,
                                                   npix,nspix,nspix_buffer);
    // -- mark active spix --
    thrust::device_vector<int> init_ids = get_unique(_spix,nbatch*npix);
    int* init_ids_ptr = thrust::raw_pointer_cast(init_ids.data());
    int nactive = init_ids.size();
    mark_active(sp_params, init_ids_ptr, nactive, nspix, nspix_buffer, sp_size);

    // -- run method --
    int max_spix = run_bist_alg(img,_spix,shifted_spix,sp_params,
                                border,sp_helper, sm_helper, sm_seg1,sm_seg2,sm_pairs,
                                niters, niters_seg, sm_start, sigma2_app, sigma2_size,
                                sp_size, potts, alpha_hastings, nspix, nspix_buffer,
                                nbatch, width, height, nftrs,params_prev,nspix_prev,
                                epsilon_reid,epsilon_new,merge_alpha,split_alpha,
                                gamma,target_nspix,prop_flag,logger);
    // printf("[after] max_spix: %d\n",max_spix);
    int prev_max_spix = max_spix;
    // printf("after.\n");
    // view_invalid(sp_params,nspix);

    // -- relabel --
    thrust::device_vector<int> prop_ids = extract_unique_ids(_spix, npix, 0);
    int nalive = prop_ids.size();

    // -- only keep superpixels which are alive --
    thrust::device_vector<int> new_ids = remove_old_ids(prop_ids,nspix_prev);
    nspix = compactify_new_superpixels(_spix,sp_params,new_ids,
                                       nspix_prev,max_spix,npix);
    //printf("final nspix: %d\n",nspix);
    // exit(1);
    // print_min_max(_spix, npix);
    // printf("3.\n");

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // -- get spixel parameters as tensors --
    // thrust::copy(_spix_ptr,_spix_ptr+npix,spix.begin());
    thrust::device_vector<int> uniq_ids = get_unique(_spix,npix);
    thrust::host_vector<int> uniq_ids_h = uniq_ids;
    // int num_ids = uniq_ids.size();
    int num_ids = nspix;
    int* _uniq_ids = thrust::raw_pointer_cast(uniq_ids.data());
    SuperpixelParams* params = get_params_as_vectors(sp_params,_uniq_ids,num_ids,nspix);
    run_update_prior(params,_uniq_ids, npix, nspix, nspix_prev,false);
    CudaFindBorderPixels_end(_spix,border,npix,nbatch,width,height);

    // -- [dev only] check nspix v.s. max --
    // printf("[@end] nspix,uniq_ids.back(),params->ids.size(): %d,%d,%d\n",
    //        nspix,uniq_ids_h.back(),params->ids.size());
    assert((nspix-1) >= uniq_ids_h.back());

    // // -- free --
    cudaFree(sm_helper);
    cudaFree(sm_pairs);
    cudaFree(sm_seg2);
    cudaFree(sm_seg1);
    cudaFree(sp_helper);
    cudaFree(sp_params);

    // -- return! --
    return std::make_tuple(_spix,border,params);
    // return std::make_tuple(_spix,border);
}

