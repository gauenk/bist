

#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "structs_3d.h"

__host__ void aos_to_soa(spix_params* aos_params, SuperpixelParams3d& soa_params);