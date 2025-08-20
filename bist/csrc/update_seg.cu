
#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif
#define THREADS_PER_BLOCK 512

#include <assert.h>
#include <stdio.h>
#include "seg_utils.h"
#include "update_seg.h"

/**********************************************
***********************************************


         Segmentation Update Kernel


***********************************************
**********************************************/


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (xmod3, ymod3) position of 3*3 block
*/

__global__
void update_seg_subset(float* img, int* seg, bool* border,
                       spix_params* sp_params,
                       const float sigma2_app, const float potts,
                       const int npix, const int nspix_buffer,
                       const int width, const int height, const int nftrs,
                       const int xmod3, const int ymod3){

    int label_check;
    int pix_idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (pix_idx>=npix)  return;
    // todo; add batch info here.
    int batch_ix = blockIdx.y;
    int label_offset = nspix_buffer*batch_ix;

    int x = pix_idx % width;  
    if (x % 2 != xmod3) return;
    int y = pix_idx / width;   
    if (y % 2 != ymod3) return;

    pix_idx = pix_idx + npix*batch_ix;
    if (border[pix_idx]==0) return;
    // strides of 2*2

    //float beta = 4;
    //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
    const bool x_greater_than_1 = (x>0);
    const bool y_greater_than_1 = (y>0);
    const bool x_smaller_than_width_minus_1 = x<(width-1);
    const bool y_smaller_than_height_minus_1 = y<(height-1);
    // if ((!x_greater_than_1)||(!y_greater_than_1)||(!x_smaller_than_width_minus_1)||(!y_smaller_than_height_minus_1)) return;
   
    bool nbrs[9];
    //float potts_term[4];
    //potts_term[0] = potts_term[1] = potts_term[2] = potts_term[3] = 0;
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 
    //printf("Potts: %f", potts);

    // -- count neighbors --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    //-- init max --
    float2 res_max;
    res_max.x = -9999;
    int C = __ldg(&seg[pix_idx]);

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (x>0){ W = __ldg(&seg[pix_idx-1]); } // left
    if (y>0){ N = __ldg(&seg[pix_idx-width]); }// top
    if (x<(width-1)){ E = __ldg(&seg[pix_idx+1]); } // right
    if (y<(height-1)){ S = __ldg(&seg[pix_idx+width]); } // below

    // --> diags [north (east, west), south (east, west)] <--
    int NE = -1, NW = -1, SE = -1, SW = -1;
    if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[pix_idx-width+1]); } // top-right
    if ((y>0) and (x>0)){  NW = __ldg(&seg[pix_idx-width-1]); } // top-left
    if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[pix_idx+width+1]); } // btm-right
    if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[pix_idx+width-1]); } // btm-left

    // -- read superpixel labels --
    // int NW =__ldg(&seg[pix_idx-width-1]);
    // int N = __ldg(&seg[pix_idx-width]);
    // int NE = __ldg(&seg[pix_idx-width+1]);
    // int W = __ldg(&seg[pix_idx-1]);
    // int E = __ldg(&seg[pix_idx+1]);
    // int SW = __ldg(&seg[pix_idx+width-1]);
    // int S = __ldg(&seg[pix_idx+width]);
    // int SE =__ldg(&seg[pix_idx+width+1]);  
    // int C = __ldg(&seg[pix_idx]);
    res_max.y = C;

    //N :
    // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d\n",
    //        NW, N, NE,  W, E, SW, S, SE);
    set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    isNvalid = nbrs[8];
    
    //W :
    set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    isWvalid = nbrs[8];

    //S :
    set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    isSvalid = nbrs[8];

    //E:
    set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    isEvalid = nbrs[8];

    // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d\n",
    //        NW, N, NE,  W, E, SW, S, SE);
    // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d | %d,%d,%d,%d\n",
    //        NW, N, NE,  W, E, SW, S, SE,
    //        isNvalid ? 1 : 0,isWvalid ? 1 : 0,
    //        isSvalid ? 1 : 0,isEvalid ? 1 : 0);
    if(!isNvalid) return;
    if(!isWvalid) return;
    if(!isSvalid) return;
    if(!isEvalid) return;
    // printf("idx: %d\n",pix_idx);

    // -- index image --
    float* imgC = img + pix_idx * 3;

    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    // assert(label_check >= 0);
    if (valid){
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check+label_offset,
                           label_check,sigma2_app,count_diff_nbrs_N,potts,res_max);
    }

    valid = S>=0;
    label_check = S;
    // assert(label_check >= 0);
    if((label_check!=N)&&valid)
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check+label_offset,
                           label_check,sigma2_app,count_diff_nbrs_S,potts,res_max);

    valid = W>=0;
    label_check = W;
    // assert(label_check >= 0);
    if ( (label_check!=S)&&(label_check!=N)&&valid)
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check+label_offset,
                           label_check,sigma2_app,count_diff_nbrs_W,potts,res_max);
    
    valid = E >=0;
    label_check = E;
    // assert(label_check >= 0);
    if((label_check!=W)&&(label_check!=S)&&(label_check!=N)&&valid)
      res_max = calc_joint(imgC,seg,x,y,sp_params,label_check+label_offset,
                           label_check,sigma2_app,count_diff_nbrs_E,potts,res_max);
    
    // printf("[b] idx: %d\n",pix_idx);
    // int count_C = sp_params[C+label_offset].count;
    // if ((count_C <= 2) and (res_max.y == C)){
    //   //printf("edge case! [%d] %d %d \n",batch_ix,C,sp_params[C+label_offset].count);
    //   if (N>=0){ res_max.y = N;}
    //   else if(E>=0){ res_max.y = E;}
    //   else if(S>=0){ res_max.y = S;}
    //   else if(W>=0){ res_max.y = W;}
    // }

    // -- update --
    assert(res_max.y>=0); // dev only
    seg[pix_idx] = res_max.y;


    return;
}



__device__ float2 calc_joint(float* imgC, int* seg,
                             int width_index, int height_index,
                             spix_params* sp_params,
			                       int seg_index, int seg_label,
			                       float sigma2_app,  float neigh_neq,
                             float beta, float2 res_max){

    // -- init res --
    float res = -1000; // some large negative number // why?
    /* float res = 0.; */

    // -- appearance --
    int count = sp_params[seg_index].count;
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_index].mu_app.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_index].mu_app.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_index].mu_app.z);
    // float sigma_app;
    // const float sigma_a = __ldg(&sp_params[seg_index].prior_sigma_app);
    // const float sigma_a_x = __ldg(&sp_params[seg_index].prior_sigma_app.x);
    // const float sigma_a_y = __ldg(&sp_params[seg_index].prior_sigma_app.y);
    // const float sigma_a_z = __ldg(&sp_params[seg_index].prior_sigma_app.z);
    // const float logdet_sigma_app = __ldg(&sp_params[seg_index].logdet_sigma_app);
    const float logdet_sigma_app = 3.*log(sigma2_app);

    // -- shape --
    const int d0 = width_index - __ldg(&sp_params[seg_index].mu_shape.x);
    const int d1 = height_index - __ldg(&sp_params[seg_index].mu_shape.y);
    const float sigma_s_x = __ldg(&sp_params[seg_index].sigma_shape.x);
    const float sigma_s_y = __ldg(&sp_params[seg_index].sigma_shape.y);
    const float sigma_s_z = __ldg(&sp_params[seg_index].sigma_shape.z);
    const float logdet_sigma_shape = __ldg(&sp_params[seg_index].logdet_sigma_shape);

    // -- appearance [sigma is actually \sigma^2] --
    res = res - (x0*x0 + x1*x1 + x2*x2)/sigma2_app;
    // printf("res: %2.2f\n",res);

    // -- info only --
    float3 mu_app = sp_params[seg_index].mu_app;
    // printf("mu_app: [%2.2f,%2.2f,%2.2f],[%2.2f,%2.2f,%2.2f].[%2.2f]\n",
    //        mu_app.x,mu_app.y,mu_app.z,imgC[0],imgC[1],imgC[2],sigma2_app);
    // res = res - logdet_sigma_app;

    // printf("mu_app: [%2.2f,%2.2f,%2.2f],[%2.2f,%2.2f,%2.2f].[%2.2f]\n",
    //        mu_app.x,mu_app.y,mu_app.z,imgC[0],imgC[1],imgC[2],sigma2_app);

    // -- shape [sigma is actually \Sigma^{(-1)}, the inverse] --
    res = res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - 2*d0*d1*sigma_s_y; // sign(s_y) = -1
    // res=res - d0*d0*sigma_s_x - d1*d1*sigma_s_z - d0*d1*sigma_s_y; // sign(s_y) = -1
    res = res - logdet_sigma_shape;

    // -- prior --
    /* res = res - sp_params[seg_index].prior_lprob; */

    // -- potts term --
    res = res - beta*neigh_neq;
    res = res - 100*(count<=10);

    // -- update res --
    if( res>res_max.x ){
      res_max.x = res;
      res_max.y = seg_label;
    }



    return res_max;
}


/**********************************************
***********************************************


              Main Function


***********************************************
**********************************************/

// __host__ void set_border(int* seg, bool* border, int height, int width){
//   int npix = height*width;
//   int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
//   dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//   dim3 BlockPerGrid(num_block,1);
//   cudaMemset(border, 0, npix*sizeof(bool));
//   find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,width,height);
// }

__host__ void update_seg(float* img, int* seg, bool* border,
                         spix_params* sp_params, const int niters,
                         const float sigma2_app, const float potts,
                         const int npix, int nspix_buffer,
			                   int nbatch, int width, int height, int nftrs,
                         Logger* logger){
    
    // printf("npix, nbatch, width, height, nftrs: %d,%d,%d,%d,%d\n",
    //        npix, nbatch, width, height, nftrs);
    // printf("1./sigma2_app: %2.2f\n",1./sigma2_app);
    int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,nbatch);
    // assert(nbatch==1);
    for (int iter = 0 ; iter < niters; iter++){
        cudaMemset(border, 0, nbatch*npix*sizeof(bool));
        find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,width,height);

        // auto opt_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        // int b_sum = torch::sum(torch::from_blob(border,{npix},opt_b).to(torch::kInt)).item<int>();
        // printf("b_sum: %d\n",b_sum);

        for (int xmod3 = 0 ; xmod3 <2; xmod3++){
            for (int ymod3 = 0; ymod3 <2; ymod3++){
                update_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
                     border, sp_params, sigma2_app, potts,\
                     npix, nspix_buffer, width, height, nftrs, xmod3, ymod3);
                  gpuErrchk( cudaPeekAtLastError() );
                  gpuErrchk( cudaDeviceSynchronize() );
                if (logger!=nullptr){
                  logger->boundary_update(seg);
                }
            }
        }
    }
    cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, npix, width, height);
}








// __global__
// void update_seg_subset_log(float* img, int* seg, bool* border,
//                            spix_params* sp_params,
//                            const float sigma2_app, const float potts,
//                            const int npix, const int nbatch,
//                            const int width, const int height, const int nftrs,
//                            const int xmod3, const int ymod3, Logger* logger){

//     int label_check;
//     int idx = threadIdx.x + blockIdx.x*blockDim.x;
//    // idx = idx_img;

//     int pix_idx = idx; 
//     if (pix_idx>=npix)  return;
//     // todo; add batch info here.

//     int x = pix_idx % width;  
//     if (x % 2 != xmod3) return;
//     int y = pix_idx / width;   
//     if (y % 2 != ymod3) return;
    
//     if (border[pix_idx]==0) return;
//     // strides of 2*2

//     //float beta = 4;
//     //printf("(%d, %d) - %d, %d, %d \n", x,y , idx_cache,threadIdx.x );
//     const bool x_greater_than_1 = (x>0);
//     const bool y_greater_than_1 = (y>0);
//     const bool x_smaller_than_width_minus_1 = x<(width-1);
//     const bool y_smaller_than_height_minus_1 = y<(height-1);
//     // if ((!x_greater_than_1)||(!y_greater_than_1)||(!x_smaller_than_width_minus_1)||(!y_smaller_than_height_minus_1)) return;
   
//     bool nbrs[9];
//     //float potts_term[4];
//     //potts_term[0] = potts_term[1] = potts_term[2] = potts_term[3] = 0;
//     bool isNvalid = 0;
//     bool isSvalid = 0;
//     bool isEvalid = 0;
//     bool isWvalid = 0; 
//     //printf("Potts: %f", potts);

//     // -- count neighbors --
//     int count_diff_nbrs_N=0;
//     int count_diff_nbrs_S=0;
//     int count_diff_nbrs_E=0;
//     int count_diff_nbrs_W=0;

//     //-- init max --
//     float2 res_max;
//     res_max.x = -9999;
//     int C = __ldg(&seg[pix_idx]);

//     // --> north, south, east, west <--
//     int N = -1, S = -1, E = -1, W = -1;
//     if (x>0){ W = __ldg(&seg[idx-1]); } // left
//     if (y>0){ N = __ldg(&seg[idx-width]); }// top
//     if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
//     if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

//     // --> diags [north (east, west), south (east, west)] <--
//     int NE = -1, NW = -1, SE = -1, SW = -1;
//     if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
//     if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
//     if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[idx+width+1]); } // btm-right
//     if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

//     // -- read superpixel labels --
//     // int NW =__ldg(&seg[pix_idx-width-1]);
//     // int N = __ldg(&seg[pix_idx-width]);
//     // int NE = __ldg(&seg[pix_idx-width+1]);
//     // int W = __ldg(&seg[pix_idx-1]);
//     // int E = __ldg(&seg[pix_idx+1]);
//     // int SW = __ldg(&seg[pix_idx+width-1]);
//     // int S = __ldg(&seg[pix_idx+width]);
//     // int SE =__ldg(&seg[pix_idx+width+1]);  
//     // int C = __ldg(&seg[pix_idx]);
//     res_max.y = C;

//     //N :
//     // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d\n",
//     //        NW, N, NE,  W, E, SW, S, SE);
//     set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, N, nbrs);
//     count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
//     isNvalid = nbrs[8];
    
//     //W :
//     set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, W, nbrs);
//     count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
//     isWvalid = nbrs[8];

//     //S :
//     set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, S, nbrs);
//     count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
//     isSvalid = nbrs[8];

//     //E:
//     set_nbrs_v1(NW, N, NE,  W, E, SW, S, SE, E, nbrs);
//     count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
//     isEvalid = nbrs[8];

//     // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d\n",
//     //        NW, N, NE,  W, E, SW, S, SE);
//     // printf("NW, N, NE,  W, E, SW, S, SE: %d,%d,%d,%d,%d,%d,%d,%d | %d,%d,%d,%d\n",
//     //        NW, N, NE,  W, E, SW, S, SE,
//     //        isNvalid ? 1 : 0,isWvalid ? 1 : 0,
//     //        isSvalid ? 1 : 0,isEvalid ? 1 : 0);
//     if(!isNvalid) return;
//     if(!isWvalid) return;
//     if(!isSvalid) return;
//     if(!isEvalid) return;
//     // printf("idx: %d\n",pix_idx);

//     // -- index image --
//     float* imgC = img + idx * 3;

//     // -- compute posterior --
//     bool valid = N >= 0;
//     label_check = N;
//     // assert(label_check >= 0);
//     if (valid){
//       res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
//                            sigma2_app,count_diff_nbrs_N,potts,res_max);
//     }

//     valid = S>=0;
//     label_check = S;
//     // assert(label_check >= 0);
//     if((label_check!=N)&&valid)
//       res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
//                            sigma2_app,count_diff_nbrs_S,potts,res_max);

//     valid = W>=0;
//     label_check = W;
//     // assert(label_check >= 0);
//     if ( (label_check!=S)&&(label_check!=N)&&valid)
//       res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
//                            sigma2_app,count_diff_nbrs_W,potts,res_max);
    
//     valid = E >=0;
//     label_check = E;
//     // assert(label_check >= 0);
//     if((label_check!=W)&&(label_check!=S)&&(label_check!=N)&&valid)
//       res_max = calc_joint(imgC,seg,x,y,sp_params,label_check,
//                            sigma2_app,count_diff_nbrs_E,potts,res_max);
    
//     // printf("[b] idx: %d\n",pix_idx);

//     int count_C = sp_params[C].count;
//     if ((count_C <= 2) and (res_max.y == C)){
//       if (N>=0){ res_max.y = N;}
//       else if(E>=0){ res_max.y = E;}
//       else if(S>=0){ res_max.y = S;}
//       else if(W>=0){ res_max.y = W;}
//     }

//     // -- update --
//     assert(res_max.y>=0); // dev only
//     seg[pix_idx] = res_max.y;


//     return;
// }



// __host__ void update_seg_log(float* img, int* seg, bool* border,
//                          spix_params* sp_params, const int niters,
//                          const float sigma2_app, const float potts,
//                              const int npix, int nbatch, int width, int height, int nftrs, Logger* logger){
    
//     // printf("npix, nbatch, width, height, nftrs: %d,%d,%d,%d,%d\n",
//     //        npix, nbatch, width, height, nftrs);
//     // printf("1./sigma2_app: %2.2f\n",1./sigma2_app);
//     int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
//     dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//     dim3 BlockPerGrid(num_block,nbatch);
//     assert(nbatch==1);
//     for (int iter = 0 ; iter < niters; iter++){
//         cudaMemset(border, 0, npix*sizeof(bool));
//         find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,npix,width,height);

//         // auto opt_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
//         // int b_sum = torch::sum(torch::from_blob(border,{npix},opt_b).to(torch::kInt)).item<int>();
//         // printf("b_sum: %d\n",b_sum);

//         for (int xmod3 = 0 ; xmod3 <2; xmod3++){
//             for (int ymod3 = 0; ymod3 <2; ymod3++){
//                 update_seg_subset_log<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, \
//                      border, sp_params, sigma2_app, potts,\
//                      npix, nbatch, width, height, nftrs, xmod3, ymod3, logger);
//                 logger->boundary_update(seg);
//             }
//         }
//     }
//     cudaMemset(border, 0, npix*sizeof(bool));
//     find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, npix, width, height);
// }



