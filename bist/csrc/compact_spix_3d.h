


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <cfloat>


uint64_t* run_compactify(uint64_t* spix, uint32_t* bids, uint64_t* nspix_old, uint64_t* max_new_nspix, int nbatch, int nnodes);