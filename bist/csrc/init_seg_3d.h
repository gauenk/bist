#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "structs.h"
#include <cfloat>



__host__ uint32_t* init_seg_3d(uint32_t* spix, float3* pos, uint8_t* bids, int* ptr, float* dim_sizes, float data_scale, int sp_size, int nbatch, int ntotal);
__global__ void InitVeronoiSeg(uint32_t* spix, uint32_t* nspix, float3* pos, uint8_t* bids, int* ptr, float* dim_sizes, float data_scale, int S, int nnodes);