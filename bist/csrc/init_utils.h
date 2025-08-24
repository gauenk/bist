/*************************************************

          This script helps allocate
          and initialize memory for
          supporting information

**************************************************/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*************************************************

               Allocation

**************************************************/

int check_cuda_error();
void throw_on_cuda_error(cudaError_t code);
void* easy_allocate(int size, int esize);
void* easy_allocate_cpu(int size, int esize);

