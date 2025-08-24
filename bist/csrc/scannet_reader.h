#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <filesystem>

#include "cuda.h"
#include "cuda_runtime.h"

struct ScanNetScene {
    std::vector<float> pos;  // XYZ positions [x1,y1,z1,x2,y2,z2,...]
    std::vector<float> ftr;  // RGB features [r1,g1,b1,r2,g2,b2,...]
    int size;                // Number of points
    float xmin, xmax, ymin, ymax, zmin, zmax;
    
    ScanNetScene();
    bool read_ply(const std::filesystem::path& scene_path);
    bool write_ply(const std::filesystem::path& scene_path,
                   const std::filesystem::path& output_root,
                   float* ftrs, float* pos, int nnodes, long* labels=nullptr);
};

// Utility functions
// int get_vertex_count(const std::string& ply_file);
int get_vertex_count(const std::filesystem::path& scene_name);

// Main function - returns (ftrs_cu, pos_cu, dim_sizes_cu, nnodes_cu, total_nodes, batchsize)
std::tuple<float3*,float3*,float3*,int*,int,int>
read_scene(const std::vector<std::filesystem::path>& scene_files);
bool write_scene(const std::vector<std::filesystem::path>& scene_files, const std::filesystem::path& output_root, 
                float3* ftrs, float3* pos, int* nnodes, int total_nodes, long* labels=nullptr);

