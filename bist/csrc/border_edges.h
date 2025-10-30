
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>


#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "structs_3d.h"

void filter_to_border_edges(PointCloudData& data);


PointCloudData get_border_data(PointCloudData& primal, PointCloudData& dual, SuperpixelParams3d& spix_params, uint32_t S, bool use_dual);
thrust::device_vector<uint32_t> get_primal_labels(PointCloudData& primal, PointCloudData& dual); // <- this doesn't really belong here...
void apply_spix_pooling(PointCloudData& data, SuperpixelParams3d& spix_params);

// std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<uint32_t>>
// get_border_edges(PointCloudData& data, SuperpixelParams3d& params, bool* border, uint32_t* edges);

// std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<int>,thrust::device_vector<uint8_t>>
// get_border_edges(uint32_t* spix, bool* border, uint32_t* edges, uint8_t* edge_batch_ids, uint32_t B, int32_t E);
std::tuple<thrust::device_vector<uint32_t>,thrust::device_vector<int>,thrust::device_vector<uint8_t>>
get_border_edges(uint32_t* spix, bool* border, uint32_t* edges, uint8_t* edge_batch_ids, uint32_t B, uint32_t E);