#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/* std::tuple<int*,float> */
/* int* */
std::tuple<int*,int*>
shift_and_fill(int* spix, SuperpixelParams* params, float* flow,
               int nbatch, int height, int width, bool prop_nc, Logger* logger=nullptr);

void animate_flow_motion(int* spix, SuperpixelParams* params, float* flow,
                         int nbatch, int height, int width, bool prop_nc, Logger* logger=nullptr);

/* std::tuple<int*,int*> */
/* shift_and_fill_log(int* spix, SuperpixelParams* params, float* flow, */
/*                    int nbatch, int height, int width, bool prop_nc, Logger* log); */


