#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs.h" // only for logger

void
run_fill_missing(int* spix, double* centers,
                 int nbatch, int height, int width, int break_iter, Logger* logger=nullptr);

__host__
void fill_missing(int* seg,  double* centers, bool* border,
                  int nbatch,  int height, int width,int break_iter, Logger* logger=nullptr);


__global__
void find_border_along_missing(const int* seg, bool* border,
                               const int nbatch, const int height,
                               const int width, int* num_neg);

__global__
void update_missing_seg_nn(int* seg, double* centers, bool* border,
                           const int nbatch, const int height, const int width,
                           const int npix, const int xmod3, const int ymod3,
                           bool print_values);

/* __global__ */
/* void update_missing_seg_nn_v2(int* seg, double* centers, bool* border, */
/*                            const int nbatch, const int height, const int width, */
/*                            const int npix, const int xmod3, const int ymod3, */
/*                               bool print_values); */
__global__
void update_missing_seg_nn_v2(int* seg, double* centers, bool* border,
                              bool* border_n, int* num_neg,
                              const int nbatch, const int height, const int width,
                              const int npix, const int xmod3, const int ymod3,
                              bool print_values);


