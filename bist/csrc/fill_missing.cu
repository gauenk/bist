
/*******************************************************

            This is just an initialization
      Fills missing pixels by spatial location alone.
            We want to get rid of any "-1"s

*******************************************************/

// -- cpp imports --
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -- local import --
#include "update_seg.h"
#include "init_utils.h"
#include "fill_missing.h"

// -- reset if needed --
#include "init_seg.h"
#include "compact_spix.h"

// -- define --
#define THREADS_PER_BLOCK 512


/**********************************************************

             -=-=-=-=- Main Function -=-=-=-=-=-

***********************************************************/

// todo: what if entire superpixel is invalid?; catch this and fill it here.

__host__
void fill_missing(int* seg,  double* centers, bool* border,
                  int nbatch, int height, int width, int break_iter,
                  int sp_size, Logger* logger){

    // -- init launch info --
    int npix = height*width;
    int num_block_sub = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block_sub,nbatch);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);

    // -- init border --
    cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    // bool* border_n = (bool*)easy_allocate(nbatch*npix,sizeof(bool));
    // cudaMemset(border_n, 0, nbatch*npix*sizeof(bool));
    const int sizeofint = sizeof(int);
    int iter = 0;
    // cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    // cudaMemset(border_n, 0, nbatch*npix*sizeof(bool));

    // -- init num neg --
    int* num_neg_gpu = (int*)easy_allocate(1, sizeof(int));
    int prev_neg = 10000000000;
    int num_neg_cpu = 1;
    int init_num_neg = -1;

    // find_border_along_missing<<<BlockPerGrid,ThreadPerBlock>>>  \
    //   (seg, border, nbatch, height, width, num_neg_gpu);
    // cudaMemcpy(border_n,border,nbatch*npix*sizeof(bool),cudaMemcpyDeviceToDevice);

    while (num_neg_cpu > 0){

      // -- early break [ for viz ] --
      if ((break_iter>0) and (iter >= break_iter)){ break; }

      //  -- find border pixels --
      cudaMemset(num_neg_gpu, 0, sizeof(int));
      cudaMemset(border, 0, nbatch*npix*sizeof(bool));
      find_border_along_missing<<<BlockPerGrid,ThreadPerBlock>>>\
        (seg, border, nbatch, height, width, num_neg_gpu);
      cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
      if (iter == 0){
        init_num_neg = num_neg_cpu;
      }

      //  -- update segmentation --
      for (int xmod3 = 0 ; xmod3 < 2; xmod3++){
        for (int ymod3 = 0; ymod3 < 2; ymod3++){
          // simple_update<<<BlockPerGrid,ThreadPerBlock>>>(\
          //   seg, centers, border, nbatch, width, height, npix, xmod3, ymod3);
          // update_missing_seg_nn_v2<<<BlockPerGrid,ThreadPerBlock>>>(\
          //  seg, centers, border, border_n, num_neg_gpu, nbatch, height, width,
          //  npix, xmod3, ymod3, false);
          update_missing_seg_nn<<<BlockPerGrid,ThreadPerBlock>>>(\
           seg, centers, border, nbatch, height, width,
           npix, xmod3, ymod3, false);
        }
      }

      if (logger != nullptr){
        logger->log_filled(seg);
      }

      // cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
      // cudaMemcpy(border,border_n,nbatch*npix*sizeof(bool),cudaMemcpyDeviceToDevice);

      // -- update previous --
      iter++;
      if ((iter>0) and (num_neg_cpu == prev_neg)){
        if (num_neg_cpu == npix){
          int _nspix = init_seg(seg, sp_size, width, height, nbatch);
          get_compact_spix(seg, npix);
          num_neg_cpu = 0;
        }else{
          auto msg = "An error of some type, the border won't shrink: %d\n";
          fprintf(stdout,msg,num_neg_cpu);
          // cv::String fname = "_fill_missing.csv";
          // save_spix_gpu(fname, seg, height, width);
          update_missing_seg_nn<<<BlockPerGrid,ThreadPerBlock>>>(         \
            seg, centers, border, nbatch, height, width, npix, 0, 0, true);
          update_missing_seg_nn<<<BlockPerGrid,ThreadPerBlock>>>(         \
            seg, centers, border, nbatch, height, width, npix, 0, 1, true);
          update_missing_seg_nn<<<BlockPerGrid,ThreadPerBlock>>>(         \
            seg, centers, border, nbatch, height, width, npix, 1, 0, true);
          update_missing_seg_nn<<<BlockPerGrid,ThreadPerBlock>>>(         \
            seg, centers, border, nbatch, height, width, npix, 1, 1, true);
          break;
          exit(1);
        }
      }
      prev_neg = num_neg_cpu;

    }

    //  -- find border pixels --
    cudaMemset(num_neg_gpu, 0, sizeof(int));
    cudaMemset(border, 0, nbatch*npix*sizeof(bool));
    find_border_along_missing<<<BlockPerGrid,ThreadPerBlock>>>\
      (seg, border, nbatch, height, width, num_neg_gpu);
    cudaMemcpy(&num_neg_cpu,num_neg_gpu,sizeof(int),cudaMemcpyDeviceToHost);
    if (num_neg_cpu > 0){
      fprintf(stdout,"negative spix exist: %d,%d,%d\n",num_neg_cpu,iter,break_iter);
    }

    // -- free memory --
    cudaFree(num_neg_gpu);
    // return init_num_neg; // number that was negative
}


/**********************************************************

             -=-=-=-=- Helper Functions -=-=-=-=-=-

***********************************************************/

__device__ inline
float3 isotropic_space(float3 res, int label, int x, int y,
                       double* center_prop, int height, int width, int dir){
  // float sim = -100;
  float dx = (1.0f*x - center_prop[0])/(1.0f*width);
  float dy = (1.0f*y - center_prop[1])/(1.0f*height);
  float sim = -dx*dx - dy*dy;
  // printf("%2.2f,%2.2f\n",sim,res.x);
  if (sim > res.x){
    res.x = sim;
    res.y = label;
    res.z = dir;
  }
  return res;
}


// __global__
// void simple_update(int* seg, double* centers, bool* border,
//                            const int nbatch, const int height, const int width,
//                            const int npix, const int xmod3, const int ymod3){   

//     // -- init --
//     int label_check;
//     int idx = threadIdx.x + blockIdx.x*blockDim.x;
//     int seg_idx = idx; 
//     if (seg_idx>=npix)  return;
//     int x = seg_idx % width;  
//     if (x % 2 != xmod3) return;
//     int y = seg_idx / width;   
//     if (y % 2 != ymod3) return;
//     if (border[seg_idx]==0) return;

//     // -- init neighbors --
//     bool nbrs[9];
//     bool isNvalid = 0;
//     bool isSvalid = 0;
//     bool isEvalid = 0;
//     bool isWvalid = 0; 

//     // -- init for now --
//     int count_diff_nbrs_N=0;
//     int count_diff_nbrs_S=0;
//     int count_diff_nbrs_E=0;
//     int count_diff_nbrs_W=0;

//     // -- init --
//     float3 res_max;
//     res_max.x = -999999;
//     res_max.y = -1;

//     // --> north, south, east, west <--
//     int N = -1, S = -1, E = -1, W = -1;
//     if (x>0){ W = __ldg(&seg[idx-1]); } // left
//     if (y>0){ N = __ldg(&seg[idx-width]); }// top
//     if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
//     if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

//     // --> diags [north (east, west), south (east, west)] <--
//     // int NE = -1, NW = -1, SE = -1, SW = -1;

//     // // -- read labels of neighbors --
//     // if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
//     // if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
//     // if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[idx+width+1]); } // btm-right
//     // if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

//     // -- read neighor labels for potts term --
//     // check 8 nbrs and save result if valid to change to the last place of array
//     // return how many nbrs different for potts term calculation
//     int min_neigh = -1;
//     int min_count = 1000;
//     bool update_min = false;

//     //N :
//     set_nbrs_v1(-1, N, -1, W, E, -1, S, -1, N, nbrs);
//     count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
//     // isNvalid = nbrs[8] or (res_max.y == -1);
//     // if(!isNvalid) return;
    
//     //E:
//     set_nbrs_v1(-1, N, -1, W, E, -1, S, -1, E, nbrs);
//     count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
//     // isEvalid = nbrs[8] or (res_max.y == -1);
//     // if(!isEvalid) return;

//     //S :
//     set_nbrs(-1, N, -1, W, E, -1, S, -1, S, nbrs);
//     count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
//     // isSvalid = nbrs[8] or (res_max.y == -1);
//     // if(!isSvalid) return;

//     //W :
//     set_nbrs(-1, N, -1, W, E, -1, S, -1, W, nbrs);
//     count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
//     // isWvalid = nbrs[8] or (res_max.y == -1);
//     // if(!isWvalid) return;

//     // -- compute posterior --
//     bool valid = N >= 0;
//     label_check = N;
//     if (valid){
//       res_max = isotropic_space(res_max, label_check, x, y,
//                                 centers+label_check*2, height, width, 0);
//       update_min = count_diff_nbrs_N < min_count;
//       min_count = update_min ? count_diff_nbrs_N : min_count;
//       min_neigh = update_min ? N : min_neigh;
//     }

//     valid = S>=0;
//     label_check = S;
//     if(valid && (label_check!=N)){
//       res_max = isotropic_space(res_max, label_check, x, y,
//                                 centers+label_check*2, height, width, 1);
//       update_min = count_diff_nbrs_S < min_count;
//       min_count = update_min ? count_diff_nbrs_S : min_count;
//       min_neigh = update_min ? S : min_neigh;
//     }

//     valid = W >= 0;
//     label_check = W;
//     if(valid && (label_check!=S)&&(label_check!=N)) {
//       res_max = isotropic_space(res_max, label_check, x, y,
//                                 centers+label_check*2, height, width, 2);
//       update_min = count_diff_nbrs_W < min_count;
//       min_count = update_min ? count_diff_nbrs_W : min_count;
//       min_neigh = update_min ? W : min_neigh;
//     }
    
//     valid = E >= 0;
//     label_check = E;
//     if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
//       res_max = isotropic_space(res_max, label_check, x, y,
//                                 centers+label_check*2, height,width, 3);
//       update_min = count_diff_nbrs_E < min_count;
//       min_count = update_min ? count_diff_nbrs_E : min_count;
//       min_neigh = update_min ? E : min_neigh;
//     }

//     // --> if we do not setting the border for a mysterious reason <--
//     //   then set it to the neighbor with the number of the same neighbors
//     if (res_max.y == -1){
//       res_max.y = min_neigh;
//     }
//     seg[seg_idx] = res_max.y;
//     return;
// }

__global__
void update_missing_seg_nn(int* seg, double* centers, bool* border,
                           const int nbatch, const int height, const int width,
                           const int npix, const int xmod3, const int ymod3,
                           bool print_values){   

    // -- init --
    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int pix_idx = idx; 
    if (pix_idx>=npix)  return;
    int x = pix_idx % width;  
    if (x % 2 != xmod3) return;
    int y = pix_idx / width;   
    if (y % 2 != ymod3) return;
    if (border[pix_idx]==0) return;
    if (seg[pix_idx]!=-1) return;

    // -- init neighbors --
    bool nbrs[9];
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 

    // -- init for now --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    // -- init --
    float3 res_max;
    res_max.x = -999999;
    res_max.y = -1;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1;
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y>0){ N = __ldg(&seg[idx-width]); }// top
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

    // --> diags [north (east, west), south (east, west)] <--
    int NE = -1, NW = -1, SE = -1, SW = -1;

    // -- read labels of neighbors --
    // if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
    // if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
    // if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[idx+width+1]); } // btm-right
    // if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

    // -- read neighor labels for potts term --
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation
    int min_neigh = -1;
    int min_count = 1000;
    bool update_min = false;

    //N :
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    // isNvalid = nbrs[8] or (res_max.y == -1);
    // if(!isNvalid) return;
    
    //E:
    // set_nbrs_v1(-1, N, -1, W, E, -1, S, -1, E, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    // isEvalid = nbrs[8] or (res_max.y == -1);
    // if(!isEvalid) return;

    //S :
    // set_nbrs(-1, N, -1, W, E, -1, S, -1, S, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    // isSvalid = nbrs[8] or (res_max.y == -1);
    // if(!isSvalid) return;

    //W :
    // set_nbrs(-1, N, -1, W, E, -1, S, -1, W, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    // isWvalid = nbrs[8] or (res_max.y == -1);
    // if(!isWvalid) return;

    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 0);
      update_min = count_diff_nbrs_N < min_count;
      min_count = update_min ? count_diff_nbrs_N : min_count;
      min_neigh = update_min ? N : min_neigh;
    }

    valid = S>=0;
    label_check = S;
    if(valid && (label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 1);
      update_min = count_diff_nbrs_S < min_count;
      min_count = update_min ? count_diff_nbrs_S : min_count;
      min_neigh = update_min ? S : min_neigh;
    }

    valid = W >= 0;
    label_check = W;
    if(valid && (label_check!=S)&&(label_check!=N)) {
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 2);
      update_min = count_diff_nbrs_W < min_count;
      min_count = update_min ? count_diff_nbrs_W : min_count;
      min_neigh = update_min ? W : min_neigh;
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height,width, 3);
      update_min = count_diff_nbrs_E < min_count;
      min_count = update_min ? count_diff_nbrs_E : min_count;
      min_neigh = update_min ? E : min_neigh;
    }


    // // -- set to diagonal if no others are valid --
    // while(res_max.y == -1){ // none are valid
    // }
    // if (print_values){
    //   double example[2];
    //   example[0] = -1;
    //   example[1] = -1;
    //   double* ex_ptr = reinterpret_cast<double*>(&example);
    //   double* cN = (N>=0) ? centers+N*2 : ex_ptr;
    //   double* cE = (E>=0) ? centers+E*2 : ex_ptr;
    //   double* cS = (S>=0) ? centers+S*2 : ex_ptr;
    //   double* cW = (W>=0) ? centers+W*2 : ex_ptr;
    //   // printf("[%d,%d] %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf)\n",x,y,N,cN[0],cN[1],E,cE[0],cE[1],S,cS[0],cS[1],W,cW[0],cW[1]);
    //   // printf("[%d,%d] %d %d %d %d (%2.3lf,%2.3lf)\n",x,y,N,E,S,W,centers[0],centers[1]);
    // }else{
    //   // printf(":D [%d,%d] %d %d %d %d (%2.3lf,%2.3lf)\n",x,y,N,E,S,W,centers[0],centers[1]);
    // }

    // --> if we do not setting the border for a mysterious reason <--
    //   then set it to the neighbor with the number of the same neighbors
    if (res_max.y == -1){
      res_max.y = min_neigh;
    }
    seg[pix_idx] = res_max.y;
    return;
}


__global__
void update_missing_seg_nn_v2(int* seg, double* centers, bool* border,
                              bool* border_n, int* num_neg,
                              const int nbatch, const int height, const int width,
                              const int npix, const int xmod3, const int ymod3,
                              bool print_values){   

    // -- init --
    int label_check;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int pix_idx = idx; 
    if (pix_idx>=npix)  return;
    int x = pix_idx % width;  
    if (x % 2 != xmod3) return;
    int y = pix_idx / width;   
    if (y % 2 != ymod3) return;
    if (border[pix_idx]==0) return;

    // -- init neighbors --
    bool nbrs[9];
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0; 

    // -- init for now --
    int count_diff_nbrs_N=0;
    int count_diff_nbrs_S=0;
    int count_diff_nbrs_E=0;
    int count_diff_nbrs_W=0;

    // -- init --
    float3 res_max;
    res_max.x = -999999;
    res_max.y = -1;
    res_max.z = -1;

    // --> north, south, east, west <--
    int C = __ldg(&seg[idx]); // center
    atomicAdd(num_neg,C==-1 ? 1 : 0 );

    int N = -1, S = -1, E = -1, W = -1;
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y>0){ N = __ldg(&seg[idx-width]); }// top
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below

    // --> diags [north (east, west), south (east, west)] <--
    int NE = -1, NW = -1, SE = -1, SW = -1;

    // -- read labels of neighbors --
    // if ((y>0) and (x<(width-1))){ NE = __ldg(&seg[idx-width+1]); } // top-right
    // if ((y>0) and (x>0)){  NW = __ldg(&seg[idx-width-1]); } // top-left
    // if ((x<(width-1)) and (y<(height-1))){SE = __ldg(&seg[idx+width+1]); } // btm-right
    // if ((x>0) and (y<(height-1))){ SW = __ldg(&seg[idx+width-1]); } // btm-left

    // -- read neighor labels for potts term --
    // check 8 nbrs and save result if valid to change to the last place of array
    // return how many nbrs different for potts term calculation
    int min_neigh = -1;
    int min_count = 1000;
    bool update_min = false;

    //N :
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, N, nbrs);
    count_diff_nbrs_N = ischangbale_by_nbrs(nbrs);
    // isNvalid = nbrs[8] or (res_max.y == -1);
    // if(!isNvalid) return;
    
    //E:
    // set_nbrs_v1(-1, N, -1, W, E, -1, S, -1, E, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, E, nbrs);
    count_diff_nbrs_E = ischangbale_by_nbrs(nbrs);
    // isEvalid = nbrs[8] or (res_max.y == -1);
    // if(!isEvalid) return;

    //S :
    // set_nbrs(-1, N, -1, W, E, -1, S, -1, S, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, S, nbrs);
    count_diff_nbrs_S = ischangbale_by_nbrs(nbrs);
    // isSvalid = nbrs[8] or (res_max.y == -1);
    // if(!isSvalid) return;

    //W :
    // set_nbrs(-1, N, -1, W, E, -1, S, -1, W, nbrs);
    set_nbrs_v1(NW, N, NE, W, E, SW, S, SE, W, nbrs);
    count_diff_nbrs_W = ischangbale_by_nbrs(nbrs);
    // isWvalid = nbrs[8] or (res_max.y == -1);
    // if(!isWvalid) return;

    // -- compute posterior --
    bool valid = N >= 0;
    label_check = N;
    if (valid){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 0);
      update_min = count_diff_nbrs_N < min_count;
      min_count = update_min ? count_diff_nbrs_N : min_count;
      min_neigh = update_min ? N : min_neigh;
    }

    valid = S>=0;
    label_check = S;
    if(valid && (label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 1);
      update_min = count_diff_nbrs_S < min_count;
      min_count = update_min ? count_diff_nbrs_S : min_count;
      min_neigh = update_min ? S : min_neigh;
    }

    valid = W >= 0;
    label_check = W;
    if(valid && (label_check!=S)&&(label_check!=N)) {
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height, width, 2);
      update_min = count_diff_nbrs_W < min_count;
      min_count = update_min ? count_diff_nbrs_W : min_count;
      min_neigh = update_min ? W : min_neigh;
    }
    
    valid = E >= 0;
    label_check = E;
    if(valid && (label_check!=W)&&(label_check!=S)&&(label_check!=N)){
      res_max = isotropic_space(res_max, label_check, x, y,
                                centers+label_check*2, height,width, 3);
      update_min = count_diff_nbrs_E < min_count;
      min_count = update_min ? count_diff_nbrs_E : min_count;
      min_neigh = update_min ? E : min_neigh;
    }


    // // -- set to diagonal if no others are valid --
    // while(res_max.y == -1){ // none are valid
    // }
    // if (print_values){
    //   double example[2];
    //   example[0] = -1;
    //   example[1] = -1;
    //   double* ex_ptr = reinterpret_cast<double*>(&example);
    //   double* cN = (N>=0) ? centers+N*2 : ex_ptr;
    //   double* cE = (E>=0) ? centers+E*2 : ex_ptr;
    //   double* cS = (S>=0) ? centers+S*2 : ex_ptr;
    //   double* cW = (W>=0) ? centers+W*2 : ex_ptr;
    //   // printf("[%d,%d] %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf) %d: (%2.3lf, %2.3lf)\n",x,y,N,cN[0],cN[1],E,cE[0],cE[1],S,cS[0],cS[1],W,cW[0],cW[1]);
    //   // printf("[%d,%d] %d %d %d %d (%2.3lf,%2.3lf)\n",x,y,N,E,S,W,centers[0],centers[1]);
    // }else{
    //   // printf(":D [%d,%d] %d %d %d %d (%2.3lf,%2.3lf)\n",x,y,N,E,S,W,centers[0],centers[1]);
    // }

    // --> if we do not setting the border for a mysterious reason <--
    //   then set it to the neighbor with the number of the same neighbors
    border_n[pix_idx] = 0;
    assert(res_max.z>=0);
    if (res_max.z == 0){
      border_n[pix_idx-width] = true;
    }else if (res_max.z == 1){
      border_n[pix_idx+width] = true;
    }else if (res_max.z == 2){
      border_n[pix_idx-1] = true;
    }else if (res_max.z == 3){
      border_n[pix_idx+1] = true;
    }
    seg[pix_idx] = res_max.y;
    // if (res_max.y == -1){
    //   res_max.y = min_neigh;
    // }
    return;
}



__global__
void find_border_along_missing(const int* seg, bool* border, 
                               const int nbatch, const int height,
                               const int width, int* num_neg){   

    // --> cuda indices <--
    int npix = height*width;
    int pix_idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (pix_idx>=npix) return; 

    // --> space coordinates <--
    int idx = pix_idx;
    int x = idx % width;
    int y = idx / width;

    // --> north, south, east, west <--
    int N = -1, S = -1, E = -1, W = -1, C = -1;

    // --> check out of bounds <--
    C = __ldg(&seg[idx]); // self
    if (y>0){ N = __ldg(&seg[idx-width]); } // above
    if (x>0){ W = __ldg(&seg[idx-1]); } // left
    if (y<(height-1)){ S = __ldg(&seg[idx+width]); } // below
    if (x<(width-1)){ E = __ldg(&seg[idx+1]); } // right
   
    // if the center is "-1" and any neighbor is valid, it is an edge
    bool valid = (N >= 0) or (W >= 0) or (S >= 0) or (E >= 0);
    valid = valid and (C==-1);
    if (valid){ border[idx]=1; }
    if (C<0){ atomicAdd(num_neg,1); }
    return;        
}


/**********************************************************

             -=-=-=-=- Python API  -=-=-=-=-=-

***********************************************************/

void run_fill_missing(int* spix, double* centers,
                      int nbatch, int height, int width,
                      int break_iter, int sp_size, Logger* logger){


    // -- allocate border --
    int npix = height*width;
    bool* border = (bool*)easy_allocate(nbatch*npix,sizeof(bool));

    // -- run fill --
    fill_missing(spix, centers, border, nbatch, height, width, break_iter, sp_size, logger);
    cudaFree(border);

    return;
}

