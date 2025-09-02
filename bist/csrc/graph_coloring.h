


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/extrema.h>


std::tuple<thrust::device_vector<uint8_t>,uint8_t> 
get_graph_coloring(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,       // CSR edge pointers (size V+1)
    int num_vertices);