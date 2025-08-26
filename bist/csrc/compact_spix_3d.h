


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <cfloat>


uint32_t* run_compactify(uint32_t* spix, uint8_t* bids, uint32_t* nspix_old, uint32_t* max_new_nspix, int nbatch, int nnodes);