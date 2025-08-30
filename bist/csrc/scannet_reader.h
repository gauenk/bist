#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <filesystem>
#include <thrust/device_vector.h>


#include "cuda.h"
#include "cuda_runtime.h"
#include "structs_3d.h"

struct ScanNetScene {
    std::vector<float> pos;  // XYZ positions [x1,y1,z1,x2,y2,z2,...]
    std::vector<float> ftr;  // RGB features [r1,g1,b1,r2,g2,b2,...]
    std::vector<uint32_t> e0;
    std::vector<uint32_t> e1;
    int size;                // Number of points
    int nfaces; // number of pairs
    float xmin, xmax, ymin, ymax, zmin, zmax;
    
    ScanNetScene();
    bool read_ply(const std::filesystem::path& scene_path);
    bool write_ply(const std::filesystem::path& scene_path,
                   const std::filesystem::path& output_root,
                   float* ftrs, float* pos, uint32_t* edges, 
                   int nnodes, int nedges, uint8_t* gcolor, uint32_t* labels=nullptr);
    bool write_spix_ply_with_fn(const std::filesystem::path& scene_path, 
                        const std::filesystem::path& output_root,
                        thrust::host_vector<float3>& ftrs, 
                        thrust::host_vector<double3>& pos,
                        thrust::host_vector<double3>& var, 
                        thrust::host_vector<double3>& cov, 
                        int nspix);
    bool write_spix_ply(const std::filesystem::path& ply_file,
                        thrust::host_vector<float3>& ftrs, 
                        thrust::host_vector<double3>& pos,
                        thrust::host_vector<double3>& var, 
                        thrust::host_vector<double3>& cov,
                        thrust::host_vector<uint32_t>& poly,
                        thrust::host_vector<uint32_t>& spix_sizes,
                        int nspix);
};

// Utility functions
// int get_vertex_count(const std::string& ply_file);
int get_vertex_count(const std::filesystem::path& scene_name);

std::vector<std::filesystem::path> get_scene_files(std::filesystem::path root);

// Main function - returns (ftrs_cu, pos_cu, dim_sizes_cu, nnodes_cu, total_nodes, batchsize)
std::tuple<float3*,float3*,uint32_t*,uint8_t*,uint8_t*,int*,int*,float*>
read_scene(const std::vector<std::filesystem::path>& scene_files);
bool write_scene(const std::vector<std::filesystem::path>& scene_files, const std::filesystem::path& output_root, 
                float3* ftrs, float3* pos, uint32_t* edges, int* ptr, int* eptr, uint8_t* gcolor, uint32_t* labels=nullptr);
bool write_spix(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, SuperpixelParams3d& spix_params);