#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int* run_shift_labels(int* pix_labels, int* spix, float* flow,
                      int* sizes, int nspix, int nbatch, int height, int width, bool overlap_sm);
