

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ inline
bool check_oob(int hi,int ix,int wi,int jx,int H,int W){
  bool oob = (hi+ix)<0 or (hi+ix)>(H-1);
  oob = oob or (wi+jx)<0 or (wi+jx)>(W-1);
  return oob;
}

/* __host__ std::tuple<torch::Tensor,torch::Tensor> */
/* run_split_disconnected(int* seg, int nbatch, int height, int width, int nspix); */
void run_invalidate_disconnected(int* seg, int nbatch, int height, int width, int nspix);

__global__
void assign_spix_to_regions(uint64_t* tmp_spix_to_regions,
                            int* region_to_spix,
                            int* region_sizes, int nregions, int nspix);
__global__
void decode_spix_to_regions(int* spix_to_regions,
                            int* spix_sizes,
                            bool* valid_regions,
                            uint64_t* tmp_spix_to_regions,
                            int nregions, int nspix);
__global__
void invalidate_disconnected_spix(int* spix, int* regions,
                                  bool* valid_regions, int npix);



__host__
std::tuple<int*,int*,int*,int> get_regions(int* seg, int B, int H, int W);

/* __host__ */
/* std::tuple<torch::Tensor,int> get_split_starts(torch::Tensor spix_counts); */

__global__
void update_spix(int* spix, int* regions, int* children,
                 int* split_starts, int* region_offsets,
                 int npix, int nspix, int max_nchild);

__global__
void compute_offsets(int* regions_spix, int* region_sizes,
                     int* region_offsets, int nspix, int nregions, int max_nchild);

__global__
void get_spix_counts(int* regions_spix, int* spix_counts, int nregions);

__global__
void relabel_spix(int* spix, int* ids, int npix, int num_ids);

__global__
void compute_region_sizes(int* size, int* regions, int npix, int nregions);

__global__
void find_spix_min(int* seg, int* curr, int* prev,
                   int* changes, int H, int W, int npix);


