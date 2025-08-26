#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cfloat>

#include "extract_edges.h"
#include "init_utils.h"


#include <vector>
#include <thrust/device_ptr.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "scannet_reader.h"

// -- read each scene --
std::tuple<float3*,float3*,uint32_t*,uint8_t*,int*,float*>
read_scene(const std::vector<std::filesystem::path>& scene_files){

    // -- read sizes --
    int batchsize = scene_files.size();
    int* ptr_cpu = (int*)malloc((batchsize+1) * sizeof(int));
    
    // First pass: get sizes
    int _ix = 1;
    ptr_cpu[0] = 0;
    for (const auto& scene_file : scene_files) {
        ptr_cpu[_ix] = get_vertex_count(scene_file)+ptr_cpu[_ix-1];
        _ix++;
    }
    int total_nodes = ptr_cpu[batchsize];
    printf("total nodes: %d\n",total_nodes);

    // Reserve space for all points
    std::vector<float> ftrs(3*total_nodes);
    std::vector<float> pos(3*total_nodes);
    std::vector<float> dim_sizes(6*batchsize);
    std::vector<uint8_t> bids(total_nodes);

    // Second pass: append data
    int _bx = 0;
    _ix = 0;
    for (const auto& scene_file : scene_files) {

        // -- .. --
        ScanNetScene scene;
        if(!scene.read_ply(scene_file)){
            exit(1);
        }

        // -- .. --
        memcpy(&ftrs[3*_ix], scene.ftr.data(), 3 * scene.size * sizeof(float));
        memcpy(&pos[3*_ix], scene.pos.data(), 3 * scene.size * sizeof(float));
        std::fill(bids.begin() + _ix, bids.begin() + _ix + scene.size, _bx);

        // -- .. --
        dim_sizes[6*_bx+0] = scene.xmin;
        dim_sizes[6*_bx+1] = scene.xmax;
        dim_sizes[6*_bx+2] = scene.ymin;
        dim_sizes[6*_bx+3] = scene.ymax;
        dim_sizes[6*_bx+4] = scene.zmin;
        dim_sizes[6*_bx+5] = scene.zmax;
        _ix += scene.size;
        _bx += 1;
    }

    // -- view --
    std::cout << "=== Scene Batch Info ===" << std::endl;
    std::cout << "Batch size: " << batchsize << std::endl;
    std::cout << "Total nodes: " << total_nodes << std::endl;

    // Print per-scene info
    int offset = 0;
    for (int i = 0; i < batchsize; ++i) {
        std::cout << "Scene " << i << ": " << (ptr_cpu[i+1] - ptr_cpu[i]) << " points" << std::endl;
        std::cout << "  Bounds: [" << dim_sizes[6*i+0] << ", " << dim_sizes[6*i+1] << "] "
                << "[" << dim_sizes[6*i+2] << ", " << dim_sizes[6*i+3] << "] "
                << "[" << dim_sizes[6*i+4] << ", " << dim_sizes[6*i+5] << "]" << std::endl;
    }

    // Print first few points
    std::cout << "First 3 points:" << std::endl;
    for (int i = 0; i < std::min(9, (int)pos.size()); i += 3) {
        std::cout << "  Pos: (" << pos[i] << ", " << pos[i+1] << ", " << pos[i+2] << ")" << std::endl;
        std::cout << "  RGB: (" << ftrs[i] << ", " << ftrs[i+1] << ", " << ftrs[i+2] << ")" << std::endl;
    }
    
    // -- copy --
    float3* ftrs_cu = (float3*)easy_allocate(total_nodes,sizeof(float3));
    float3* pos_cu = (float3*)easy_allocate(total_nodes,sizeof(float3));
    uint32_t* edges_cu = nullptr;
    uint8_t* bids_cu = (uint8_t*)easy_allocate(total_nodes,sizeof(uint8_t));
    int* ptr_cu = (int*)easy_allocate(batchsize+1,sizeof(int));
    float* dim_sizes_cu = (float*)easy_allocate(6*batchsize,sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(ftrs_cu,thrust::raw_pointer_cast(ftrs.data()),total_nodes*sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy(pos_cu,thrust::raw_pointer_cast(pos.data()),total_nodes*sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy(bids_cu,thrust::raw_pointer_cast(bids.data()),total_nodes*sizeof(uint8_t),cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_cu,ptr_cpu,(batchsize+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dim_sizes_cu,thrust::raw_pointer_cast(dim_sizes.data()),6*batchsize*sizeof(float),cudaMemcpyHostToDevice);
    return std::tuple(ftrs_cu,pos_cu,edges_cu,bids_cu,ptr_cu,dim_sizes_cu);

}

// -- write each scene; [nnodes == spix if point-cloud is the superpixel point cloud] --
bool write_scene(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, 
                float3* ftrs_cu, float3* pos_cu, int* ptr_cu, uint32_t* labels_cu){

    // -- sync before io --
    cudaDeviceSynchronize();

    // -- read nnodes --
    int nbatch = scene_files.size();
    int nnodes;
    cudaMemcpy(&nnodes,&ptr_cu[nbatch],sizeof(int),cudaMemcpyDeviceToHost);
    printf("nnodes: %d\n",nnodes);
    
    // -- allocate --
    float* ftrs = (float*)malloc(3*nnodes*sizeof(float));
    float* pos = (float*)malloc(3*nnodes*sizeof(float));
    //float* dim_sizes = (float*)malloc(6*nbatch,sizeof(float));
    int* ptr = (int*)malloc((nbatch+1)*sizeof(int));
    uint32_t* labels = nullptr;
    if (labels_cu != nullptr){
        labels = (uint32_t*)malloc(nnodes*sizeof(uint32_t));
    }

    // -- read to cpu --
    cudaMemcpy(ftrs,ftrs_cu,nnodes*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(pos,pos_cu,nnodes*sizeof(float3),cudaMemcpyDeviceToHost);
    //cudaMemcpy(dim_sizes,dim_sizes_cu,2*nbatch*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr,ptr_cu,(nbatch+1)*sizeof(int),cudaMemcpyDeviceToHost);
    if (labels_cu != nullptr){
        cudaMemcpy(labels,labels_cu,nnodes*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    // Second pass: append data
    // float* ftrs_b = ftrs;
    // float* pos_b = pos;
    // int* ptr_b = ptr;
    int ix = 0;
    for (const auto& scene_file : scene_files) {
        
        // -- get pointers --
        float* ftrs_b = &ftrs[3*ptr[ix]];
        float* pos_b = &pos[3*ptr[ix]];
        uint32_t* labels_b = (labels != nullptr) ? &labels[ptr[ix]] : nullptr;
        int nnodes = ptr[ix+1] - ptr[ix];
        if (labels!=nullptr){
            printf("labels_b: %ld\n",labels_b[0]);
        }

        // -- .. --
        ScanNetScene scene;
        if(!scene.write_ply(scene_file,output_root,ftrs_b,pos_b,nnodes,labels_b)){
            exit(1);
        }

        ix += 1;
    }


    free(ftrs);
    free(pos);
    free(ptr);
    if (labels != nullptr){
        free(labels);
    }
    //free(dim_sizes);
    return 0;
}

int get_vertex_count(const std::filesystem::path& scene_path) {

    // -- get filenames --
    std::string scene_name = scene_path.filename().string();
    std::filesystem::path ply_file = scene_path / (scene_name + "_vh_clean_2.ply");
    std::ifstream file(ply_file.string());
    if (!file.is_open()) {
        printf("didn't get the vertex count!\n");
        return -1;
    }

    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            return std::stoi(line.substr(15));
        }
        if (line == "end_header") break;  // Stop if we hit end of header
    }
    return -1;
}

ScanNetScene::ScanNetScene() : size(0), 
    xmin(FLT_MAX), xmax(-FLT_MAX), 
    ymin(FLT_MAX), ymax(-FLT_MAX),
    zmin(FLT_MAX), zmax(-FLT_MAX) {}
    
bool ScanNetScene::read_ply(const std::filesystem::path& scene_path) {

    // -- helper --
    std::string line;

    // -- get filenames --
    std::string scene_name = scene_path.filename().string();
    std::filesystem::path info_file = scene_path / (scene_name + ".txt");
    std::filesystem::path ply_file = scene_path / (scene_name + "_vh_clean_2.ply");
    std::cout << info_file << std::endl;
    std::cout << ply_file << std::endl;

    // -- read the axis alignment matrix --
    std::ifstream info_stream(info_file);
    float axis_align[16];
    while (std::getline(info_stream, line)) {
        if (line.rfind("axisAlignment", 0) == 0) { // starts with "axisAlignment"
            std::istringstream iss(line.substr(line.find('=') + 1));
            for (int i = 0; i < 16; i++) {
                if (!(iss >> axis_align[i])) {
                    std::cerr << "Error parsing axisAlignment\n";
                    return 1;
                }
            }
            break;
        }
    }

    // // Print result to verify
    // for (int i = 0; i < 16; i++) {
    //     std::cout << axis_align[i] << (i % 4 == 3 ? "\n" : " ");
    // }

    // -- read ply file --
    std::cout << ply_file.string() << std::endl;
    std::ifstream file(ply_file.string(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Can not open: " << ply_file << std::endl;
        return false;
    }
    
    // Parse header
    int vertex_count = 0;
    bool binary_format = false;
    
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
        if (line.find("element vertex") != std::string::npos) {
            vertex_count = std::stoi(line.substr(15));
        } else if (line.find("format binary") != std::string::npos) {
            binary_format = true;
        } else if (line == "end_header") {
            break;
        }
    }
    printf("vertex count: %d\n",vertex_count);

    // Allocate vectors
    pos.reserve(3*vertex_count);
    ftr.reserve(3*vertex_count);
    
    // Read data
    if (binary_format) {
        for (int i = 0; i < vertex_count; ++i) {
            float x, y, z;
            unsigned char r, g, b, alpha;
            
            // -- read --
            file.read(reinterpret_cast<char*>(&x), sizeof(float));
            file.read(reinterpret_cast<char*>(&y), sizeof(float));
            file.read(reinterpret_cast<char*>(&z), sizeof(float));
            file.read(reinterpret_cast<char*>(&r), sizeof(unsigned char));
            file.read(reinterpret_cast<char*>(&g), sizeof(unsigned char));
            file.read(reinterpret_cast<char*>(&b), sizeof(unsigned char));
            file.read(reinterpret_cast<char*>(&alpha), sizeof(unsigned char));

            // -- axis align --
            float x_new = axis_align[0]*x + axis_align[1]*y + axis_align[2]*z + axis_align[3];
            float y_new = axis_align[4]*x + axis_align[5]*y + axis_align[6]*z + axis_align[7];
            float z_new = axis_align[8]*x + axis_align[9]*y + axis_align[10]*z + axis_align[11];

            // -- update --
            x = x_new;
            y = y_new;
            z = z_new;

            // -- update bounds --
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
            
            // -- append --
            pos[3*i+0] = x;
            pos[3*i+1] = y;
            pos[3*i+2] = z;
            ftr[3*i+0] = r / 255.0f;
            ftr[3*i+1] = g / 255.0f;
            ftr[3*i+2] = b / 255.0f;
        }
    } else {
        for (int i = 0; i < vertex_count; ++i) {
            // -- read --
            float x, y, z;
            int r, g, b, alpha;
            file >> x >> y >> z >> r >> g >> b >> alpha;
            //printf("x,y,z: %.2f %.2f %.2f\n",x,y,z);
            
            // -- axis align --
            float x_new = axis_align[0]*x + axis_align[1]*y + axis_align[2]*z + axis_align[3];
            float y_new = axis_align[4]*x + axis_align[5]*y + axis_align[6]*z + axis_align[7];
            float z_new = axis_align[8]*x + axis_align[9]*y + axis_align[10]*z + axis_align[11];

            // -- update --
            x = x_new;
            y = y_new;
            z = z_new;

            // -- update bounds --
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);

            // -- append --
            pos[3*i+0] = x;
            pos[3*i+1] = y;
            pos[3*i+2] = z;
            ftr[3*i+0] = r / 255.0f;
            ftr[3*i+1] = g / 255.0f;
            ftr[3*i+2] = b / 255.0f;
        }
    }
    
    size = vertex_count;
    file.close();
    return true;
};

bool ScanNetScene::write_ply(const std::filesystem::path& scene_path, 
                            const std::filesystem::path& output_root,
                            float* ftrs, float* pos, int nnodes, uint32_t* labels) {

    // -- helper --
    std::string line;

    // -- make dir if needed --
    std::string scene_name = scene_path.filename().string();
    std::filesystem::path write_path = output_root / scene_name;
    if (!std::filesystem::exists(write_path)) {
        std::filesystem::create_directories(write_path);
    }

    // -- get filenames --
    std::filesystem::path ply_file = write_path / (scene_name + "_vh_clean_2.ply");
    std::cout << ply_file << std::endl;
    
    // -- delete existing file if it exists --
    if (std::filesystem::exists(ply_file)) {
        std::filesystem::remove(ply_file);
    }

    // -- open file for writing --
    std::ofstream file(ply_file.string(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Can not open: " << ply_file.string() << std::endl;
        return false;
    }

    // -- write PLY header --
    std::string header = "ply\n";
    header += "format binary_little_endian 1.0\n";
    header += "comment MLIB generated\n";
    header += "element vertex " + std::to_string(nnodes) + "\n";
    header += "property float x\n";
    header += "property float y\n";
    header += "property float z\n";
    header += "property uchar red\n";
    header += "property uchar green\n";
    header += "property uchar blue\n";
    header += "property uchar alpha\n";
    if (labels != nullptr) {
        header += "property uint label\n";
    }
    header += "end_header\n";
    file.write(header.c_str(), header.length());

    // -- write data --
    for (int i = 0; i < nnodes; ++i) {
        
        // -- init --
        float x, y, z;
        unsigned char r, g, b, alpha;
        uint32_t label;

        // -- unpack --
        x = pos[3*i+0];
        y = pos[3*i+1];
        z = pos[3*i+2];
        r = static_cast<unsigned char>(ftrs[3*i+0] * 255.0f);
        g = static_cast<unsigned char>(ftrs[3*i+1] * 255.0f);
        b = static_cast<unsigned char>(ftrs[3*i+2] * 255.0f);
        alpha = 255;
        label = (labels != nullptr) ? static_cast<uint32_t>(labels[i]) : 0;
        //printf("label: %ld\n",label);
        //printf("x,y,z r,g,b: %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n",x,y,z,ftrs[3*i+0],ftrs[3*i+1] ,ftrs[3*i+2] );


        // -- write --
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));
        // -- write label if provided --
        if (labels != nullptr) {
            file.write(reinterpret_cast<const char*>(&label), sizeof(uint32_t));
        }

    }
    
    file.close();
    return true;

};

// // Usage
// int main() {
//     ScanNetScene scene;
    
//     if (scene.load_from_ply("/path/to/scene0000_00_vh_clean_2.ply")) {
//         std::cout << "Loaded " << scene.size << " points" << std::endl;
//         std::cout << "Position vector size: " << scene.pos.size() << std::endl;
//         std::cout << "Feature vector size: " << scene.ftr.size() << std::endl;
        
//         // Now you can copy to CUDA:
//         // cudaMemcpy(cuda_pos, scene.pos.data(), scene.size * sizeof(float3), cudaMemcpyHostToDevice);
//         // cudaMemcpy(cuda_ftr, scene.ftr.data(), scene.size * sizeof(float3), cudaMemcpyHostToDevice);
//     }
    
//     return 0;
// }