

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


// std::tuple<uint32_t*, uint32_t*>
// get_csr_graph_from_edges(uint32_t* edges, uint8_t* ebids, int* eptr, int* vptr, int V_total, int E_total);


std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint32_t>>
get_csr_graph_from_edges(uint32_t* edges, uint8_t* ebids, int* eptr, int* vptr, int V_total, int E_total);


//std::tuple<uint32_t*, int*>
std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<int>,thrust::device_vector<uint8_t>>
get_edges_from_csr(uint32_t* csr_edges, uint32_t* csr_eptr, int* vptr, uint8_t* vbids, int V, int B);