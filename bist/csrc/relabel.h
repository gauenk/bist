
// -- thrust --
#include <thrust/device_vector.h>

// -- project --
#include "structs.h"


__global__
void relabel_spix_kernel(int* spix, bool* relabel, int* relabel_id,
                         int npix, int nspix, int nspix_prev);

__global__
void relabel_living_kernel(int* living_ids, bool* relabel,
                           int* relabel_id, int nliving, int nspix);

__global__
void mark_for_relabel(bool* relabel,
                      int* relabel_id,
                      uint64_t* comparisons,
                      float* ss_comps,
                      bool* is_living, int* max_spix,
                      int nspix, int nspix_prev,
                      float epsilon_reid, float epsilon_new);

__global__
void find_most_similar_spix(uint64_t* comparisons,
                            float* ss_comps,
                            spix_params* params,
                            float* mu_app_prior,
                            double* mu_shape_prior,
                            bool* is_living,
                            int height, int width,
                            int nspix, int nspix_prev, int ntotal);

int relabel_spix(int* spix, spix_params* sp_params,
                 SuperpixelParams* params_prev,
                 thrust::device_vector<int>& living_ids,
                 float epsilon_reid, float epsilon_new,
                 int height, int width, int nspix_prev, int _max_spix,Logger* logger=nullptr);

