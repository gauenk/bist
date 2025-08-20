/********************************************************************

      A split-merge loop designed for multi-scale superpixels

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

// -- primary functions --
#include "split_merge_prop.h"
#include "update_params.h"
#include "update_seg.h"

#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/


__host__ int run_smloop_alg(float* img, int* seg,
                            spix_params* sp_params, bool* border,
                            spix_helper* sp_helper, spix_helper_sm_v2* sm_helper,
                            int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
                            int niters, float sigma2_app, int sp_size, float potts,
                            float alpha_hastings, int nspix, int nspix_buffer,
                            int nbatch, int width, int height, int nftrs,
                            float merge_alpha, float split_alpha, Logger* logger){

  
    // -- fixed; not input --
    int gamma = 0;
    float sigma2_size = 0.0;
    bool prop_flag = false;
    int niters_seg = 4;
    printf("alpha: %2.3f\n",alpha_hastings);
    printf("niters: %d\n",niters);

    // -- init --
    int count = 1;
    int npix = height * width;
    int max_spix = nspix-1;
    int* shifted = seg;

    for (int idx = 0; idx < niters; idx++) {

      // -- Update Parameters --
      update_params(img, seg, sp_params, sp_helper, sigma2_app,
                    npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

      // -- Update Border --
      set_border(seg, border, height, width);


      // // -- Run Split/Merge --
      // if(idx%4 == 0){
      //   // count = 2;

        // max_spix = run_split_p(img, seg, shifted, border, sp_params,
        //                        sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
        //                        alpha_hastings, split_alpha, gamma,
        //                        sigma2_app, sigma2_size,
        //                        count, idx, max_spix,
        //                        sp_size,npix,nbatch,width,
        //                        height,nftrs,nspix_buffer,logger);


        // // -- Update Parameters --
        // update_params(img, seg, sp_params, sp_helper, sigma2_app,
        //               npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

      // }
      // if( idx%4 == 1){
        run_merge_p(img, seg, border, sp_params,
                    sp_helper, sm_helper, sm_seg1, sm_seg2, sm_pairs,
                    merge_alpha,alpha_hastings, sigma2_app, sigma2_size,count,idx,
                    max_spix,sp_size,npix,nbatch,width,height,nftrs,nspix_buffer,logger);

    }
    //     // // -- Update Parameters --
    //     // update_params(img, seg, sp_params, sp_helper, sigma2_app,
    //     //               npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

    //   }

    // }

    // // -- update params [idk; probably delete me] --
    // update_params(img, seg, sp_params, sp_helper, sigma2_app,
    //               npix, sp_size, nspix_buffer, nbatch, width, nftrs, prop_flag);

    CudaFindBorderPixels_end(seg, border, npix, nbatch, width, height);
    return max_spix;

}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

int* run_smloop(float* img, int* init_spix, int init_nspix,
                int nbatch, int height, int width, int nftrs,
                int niters, int sp_size, float sigma2_app,
                float potts, float alpha_hastings, 
                float merge_alpha, float split_alpha, Logger* logger){


    // -- unpack --
    int npix = height*width;
    assert(nbatch==1);    
    
    // -- allocate filled spix --
    int nspix = init_nspix;
    int* _spix = (int*)easy_allocate(nbatch*npix,sizeof(int));
    cudaMemcpy(_spix, init_spix, npix*sizeof(int), cudaMemcpyDeviceToDevice);
    //printf("nspix_prev: %d\n",nspix_prev);

    // // -- allocate memory --
    int nspix_buffer = init_nspix*5;
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
    //             Run Split-Merge Loop
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- init params --
    spix_params* sp_params=(spix_params*)easy_allocate(nspix_buffer,sparam_size);

    // -- mark active spix --
    thrust::device_vector<int> init_ids = get_unique(_spix,nbatch*npix);
    int* init_ids_ptr = thrust::raw_pointer_cast(init_ids.data());
    int nactive = init_ids.size();
    // mark_active(sp_params, init_ids_ptr, nactive, nspix, nspix_buffer, sp_size);
    mark_active_contiguous(sp_params,nspix,nspix_buffer,sp_size);
    init_sp_params(sp_params,sigma2_app,img,_spix,sp_helper,
                   npix,nspix,nspix_buffer,nbatch,width,nftrs,sp_size);

    // -- run method --
    int max_spix = run_smloop_alg(img,_spix,sp_params,border,
                                  sp_helper, sm_helper, sm_seg1,sm_seg2,sm_pairs,
                                  niters, sigma2_app, sp_size,
                                  potts, alpha_hastings,
                                  nspix, nspix_buffer,
                                  nbatch, width, height, nftrs,
                                  merge_alpha,split_alpha,logger);
    int prev_max_spix = max_spix;

    // -- only keep superpixels which are alive --
    thrust::device_vector<int> prop_ids = extract_unique_ids(_spix, npix, 0);
    thrust::device_vector<int> new_ids = remove_old_ids(prop_ids,init_nspix);
    nspix = compactify_new_superpixels(_spix,sp_params,new_ids,
                                       init_nspix,max_spix,npix);

    // // -- free --
    cudaFree(sm_helper);
    cudaFree(sm_pairs);
    cudaFree(sm_seg2);
    cudaFree(sm_seg1);
    cudaFree(sp_helper);
    cudaFree(sp_params);
    cudaFree(border);

    // -- return! --
    return _spix;

}

