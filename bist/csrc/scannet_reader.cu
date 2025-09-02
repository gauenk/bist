#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cfloat>

#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


#include "cuda.h"
#include "cuda_runtime.h"


#include "extract_edges.h"
#include "init_utils.h"
#include "scannet_reader.h"
#include "structs_3d.h"


std::vector<std::filesystem::path> get_scene_files(std::filesystem::path root) {
   std::vector<std::filesystem::path> scene_files;
   for (const auto& entry : std::filesystem::directory_iterator(root)) {
       if (entry.is_directory()) {
           scene_files.push_back(entry.path());
       }
   }
   return scene_files;
}


PointCloudData read_scene(const std::vector<std::filesystem::path>& scene_files) {
    
    // Initialize batch processing
    const int batch_size = scene_files.size();
    printf("Processing %d scene files\n", batch_size);

    // Initialize offset arrays for batching
    std::vector<int> vertex_ptr(batch_size + 1, 0);
    std::vector<int> edge_ptr(batch_size + 1, 0);
    std::vector<int> face_ptr(batch_size + 1, 0);
    
    // First pass: Calculate cumulative sizes for memory allocation
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& scene_file = scene_files[batch_idx];
        vertex_ptr[batch_idx + 1] = vertex_ptr[batch_idx] + get_vertex_count(scene_file);
        face_ptr[batch_idx + 1] = face_ptr[batch_idx] + get_face_count(scene_file);
    }
    
    const int total_vertices = vertex_ptr[batch_size];
    const int total_faces = face_ptr[batch_size];
    printf("Total vertices: %d, Total faces: %d\n", total_vertices, total_faces);

    // Allocate host memory for all data
    std::vector<float3> features(total_vertices);
    std::vector<float3> positions(total_vertices);
    std::vector<float> bounding_boxes(6 * batch_size, 0.0f);  // xmin,xmax,ymin,ymax,zmin,zmax per scene
    std::vector<uint8_t> vertex_batch_ids(total_vertices, 0);
    std::vector<uint32_t> faces(3 * total_faces, 0);
    
    // Edge data (stored on device)
    thrust::device_vector<uint32_t> all_edges;
    std::vector<uint8_t> edge_batch_ids;

    // Second pass: Load and copy scene data
    int vertex_offset = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& scene_file = scene_files[batch_idx];
        
        // Load scene data
        ScanNetScene scene;
        if (!scene.read_ply(scene_file)) {
            fprintf(stderr, "Failed to read scene file: %s\n", scene_file.c_str());
            exit(1);
        }

        // Copy vertex features and positions
        const int scene_vertex_count = scene.size;
        memcpy(&features[vertex_offset], scene.ftr.data(), scene_vertex_count * sizeof(float3));
        memcpy(&positions[vertex_offset], scene.pos.data(), scene_vertex_count * sizeof(float3));
        
        // Set batch IDs for this scene's vertices
        std::fill(vertex_batch_ids.begin() + vertex_offset, 
                  vertex_batch_ids.begin() + vertex_offset + scene_vertex_count, 
                  batch_idx);

        // Copy face data
        const int scene_face_count = face_ptr[batch_idx + 1] - face_ptr[batch_idx];
        const int face_offset = face_ptr[batch_idx];
        memcpy(&faces[3 * face_offset], scene.faces.data(), 3 * scene_face_count * sizeof(uint32_t));

        // Extract and append edges
        thrust::device_vector<uint32_t> scene_edges = extract_edges_from_pairs(scene.e0, scene.e1);
        const size_t scene_edge_count = scene_edges.size() / 2;  // Each edge has 2 vertices
        
        all_edges.insert(all_edges.end(), scene_edges.begin(), scene_edges.end());
        edge_batch_ids.insert(edge_batch_ids.end(), scene_edge_count, batch_idx);
        
        edge_ptr[batch_idx + 1] = edge_ptr[batch_idx] + scene_edge_count;
        printf("Scene %d edges: %d (cumulative: %d)\n", 
               batch_idx, (int)scene_edge_count, edge_ptr[batch_idx + 1]);

        // Store bounding box information
        const int bbox_offset = 6 * batch_idx;
        bounding_boxes[bbox_offset + 0] = scene.xmin;
        bounding_boxes[bbox_offset + 1] = scene.xmax;
        bounding_boxes[bbox_offset + 2] = scene.ymin;
        bounding_boxes[bbox_offset + 3] = scene.ymax;
        bounding_boxes[bbox_offset + 4] = scene.zmin;
        bounding_boxes[bbox_offset + 5] = scene.zmax;
        
        vertex_offset += scene_vertex_count;
    }

    // Print summary information
    // -- view --
    std::cout << "=== Scene Batch Info ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Total vertices: " << total_vertices << std::endl;

    // Print per-scene info
    for (int i = 0; i < batch_size; ++i) {
        std::cout << "Scene " << i << ": " << (vertex_ptr[i+1] - vertex_ptr[i]) << " points" << std::endl;
        std::cout << "  Bounds: [" << bounding_boxes[6*i+0] << ", " << bounding_boxes[6*i+1] << "] "
                << "[" << bounding_boxes[6*i+2] << ", " << bounding_boxes[6*i+3] << "] "
                << "[" << bounding_boxes[6*i+4] << ", " << bounding_boxes[6*i+5] << "]" << std::endl;
    }

    // Print first few points
    std::cout << "First 3 points:" << std::endl;
    for (int i = 0; i < std::min(3, total_vertices); i++) {
        std::cout << "  Pos: (" << positions[i].x << ", " << positions[i].y << ", " << positions[i].z << ")" << std::endl;
        std::cout << "  RGB: (" << features[i].x << ", " << features[i].y << ", " << features[i].z << ")" << std::endl;
    }

    // Create and return result
    const int total_edges = edge_ptr[batch_size];
    printf("Total edges: %d (edge batch IDs: %zu)\n", total_edges, edge_batch_ids.size());
    
    // Note: These appear to be unused in the original code
    thrust::device_vector<uint8_t> unused_gcolors;
    thrust::device_vector<uint32_t> unused_csr_edges;
    thrust::device_vector<uint32_t> unused_csr_eptr;

    return PointCloudData(
        features, positions, faces, all_edges,
        vertex_batch_ids, edge_batch_ids, vertex_ptr, edge_ptr, face_ptr,
        bounding_boxes, 0, batch_size, total_vertices, total_edges, total_faces
    );
}



// -- write each scene; [nnodes == spix if point-cloud is the superpixel point cloud] --
bool write_scene(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, PointCloudData& data){
                // float3* ftrs_cu, float3* pos_cu, uint32_t* edges_cu, int* ptr_cu, int* eptr_cu, 
                // uint8_t* gcolor_cu, uint32_t* labels_cu){

    // -- sync before io --
    cudaDeviceSynchronize();

    // // -- read nnodes --
    // int nbatch = scene_files.size();
    // int nnodes;
    // cudaMemcpy(&nnodes,&ptr_cu[nbatch],sizeof(int),cudaMemcpyDeviceToHost);
    // int nedges;
    // cudaMemcpy(&nedges,&eptr_cu[nbatch],sizeof(int),cudaMemcpyDeviceToHost);
    // printf("nnodes: %d\n",nnodes);
    // printf("nedges: %d\n",nedges);
    
    // // -- allocate --
    // float* ftrs = (float*)malloc(3*nnodes*sizeof(float));
    // float* pos = (float*)malloc(3*nnodes*sizeof(float));
    // uint32_t* edges = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
    // //float* dim_sizes = (float*)malloc(6*nbatch,sizeof(float));
    // int* ptr = (int*)malloc((nbatch+1)*sizeof(int));
    // int* eptr = (int*)malloc((nbatch+1)*sizeof(int));
    // uint8_t* gcolor = (uint8_t*)malloc(nnodes*sizeof(uint8_t));
    // uint32_t* labels = nullptr;
    // if (labels_cu != nullptr){
    //     labels = (uint32_t*)malloc(nnodes*sizeof(uint32_t));
    // }

    // // -- read to cpu --
    // cudaMemcpy(ftrs,ftrs_cu,nnodes*sizeof(float3),cudaMemcpyDeviceToHost);
    // cudaMemcpy(pos,pos_cu,nnodes*sizeof(float3),cudaMemcpyDeviceToHost);
    // cudaMemcpy(edges,edges_cu,2*nedges*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    // //cudaMemcpy(dim_sizes,dim_sizes_cu,2*nbatch*sizeof(float3),cudaMemcpyDeviceToHost);
    // cudaMemcpy(ptr,ptr_cu,(nbatch+1)*sizeof(int),cudaMemcpyDeviceToHost);
    // cudaMemcpy(eptr,eptr_cu,(nbatch+1)*sizeof(int),cudaMemcpyDeviceToHost);
    // cudaMemcpy(gcolor,gcolor_cu,nnodes*sizeof(uint8_t),cudaMemcpyDeviceToHost);
    // if (labels_cu != nullptr){
    //     cudaMemcpy(labels,labels_cu,nnodes*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    // }
    // cudaDeviceSynchronize();
    // Second pass: append data
    // float* ftrs_b = ftrs;
    // float* pos_b = pos;
    // int* ptr_b = ptr;
    for(int batch_index=0; batch_index < data.B; batch_index++){
    //for (const auto& scene_file : scene_files) {
        
        // // -- get pointers --
        // float3* ftrs_b = &ftrs[3*ptr[ix]];
        // float3* pos_b = &pos[3*ptr[ix]];
        // uint32_t* edges_b = &edges[2*eptr[ix]];
        // uint8_t*  gcolor_b = &gcolor[ptr[ix]];
        // uint32_t* labels_b = (labels != nullptr) ? &labels[ptr[ix]] : nullptr;
        // int nnodes = ptr[ix+1] - ptr[ix];
        // int nedges = eptr[ix+1] - eptr[ix];
        // printf("nedges: %d\n",nedges);
        // if (labels!=nullptr){
        //     printf("labels_b: %ld\n",labels_b[0]);
        // }

        // -- write original data (for dev) --
        auto& scene_file = scene_files[batch_index];
        ScanNetScene scene;
        PointCloudDataHost host_data(data,batch_index);
        for (int ix = 0; ix< 10; ix++){
            printf("ftrs: %2.2f %2.2f %2.2f\n",host_data.ftrs[ix].x,host_data.ftrs[ix].y,host_data.ftrs[ix].z);
        }
        // if(!scene.write_ply_with_fn(scene_file,output_root,ftrs_b,pos_b,edges_b,nnodes,nedges,gcolor_b,labels_b)){
        if(!scene.write_ply_with_fn(scene_file,output_root,host_data)){

            exit(1);
        }

        // ix += 1;
    }


    // free(ftrs);
    // free(pos);
    // free(edges);
    // free(ptr);
    // free(eptr);
    // free(gcolor);
    // if (labels != nullptr){
    //     free(labels);
    // }
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

int get_face_count(const std::filesystem::path& scene_path) {

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
        if (line.find("element face") != std::string::npos) {
            return std::stoi(line.substr(13));
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
    float axis_align[16];
    for (int index = 0; index < 16; index++) {
        int i = index / 4;
        int j = index % 4;
        axis_align[index] = 1.0 * (i==j);
    }
    if (std::filesystem::exists(info_file)){
        std::ifstream info_stream(info_file);
        while (std::getline(info_stream, line)) {
            if (line.rfind("axisAlignment", 0) == 0) { // starts with "axisAlignment"
                std::istringstream iss(line.substr(line.find('=') + 1));
                for (int i = 0; i < 16; i++) {
                    if (!(iss >> axis_align[i])) {
                        std::cerr << "Error parsing axisAlignment\n";
                        return false;
                    }
                }
                break;
            }
        }
        info_stream.close();
    }

    // -- read ply file --
    std::cout << ply_file.string() << std::endl;
    std::ifstream file(ply_file.string(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Can not open: " << ply_file << std::endl;
        return false;
    }
    
    // Parse header
    int vertex_count = 0;
    int face_count = 0;
    bool binary_format = false;
    
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
        if (line.find("element vertex") != std::string::npos) {
            vertex_count = std::stoi(line.substr(15));
        } else if (line.find("element face") != std::string::npos) {
            face_count = std::stoi(line.substr(13));
        } else if (line.find("format binary") != std::string::npos) {
            binary_format = true;
        } else if (line == "end_header") {
            break;
        }
    }
    printf("vertex count: %d\n",vertex_count);
    printf("face count: %d\n",face_count);

    // Allocate vectors - now using proper sizing
    pos.resize(vertex_count);
    ftr.resize(vertex_count);
    e0.resize(3*face_count); // triangles
    e1.resize(3*face_count);
    faces.resize(3*face_count);

    // Initialize bounds
    xmin = ymin = zmin = std::numeric_limits<float>::max();
    xmax = ymax = zmax = std::numeric_limits<float>::lowest();
    
    // Read data
    int _ei = 0;
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
            float3 pos_i = make_float3(x,y,z);
            pos[i] = pos_i;
            float3 ftr_i = make_float3(r/255.0f,g/255.0f,b/255.0f);
            // if (i < 10){
            //     printf("%2.2f %2.2f %2.2f\n",ftr_i.x,ftr_i.y,ftr_i.z);
            // }else{
            //     exit(1);
            // }
            ftr[i] = ftr_i;
        }

        // -- read faces and extract edges --
        for (int i = 0; i < face_count; ++i) {
            unsigned char vertex_count_face;
            file.read(reinterpret_cast<char*>(&vertex_count_face), sizeof(unsigned char));
            if (vertex_count_face != 3){
                printf("vertex count: %d\n",vertex_count_face);
            }
            std::vector<int> vertices(vertex_count_face);
            for (int j = 0; j < vertex_count_face; ++j) {
                file.read(reinterpret_cast<char*>(&vertices[j]), sizeof(int));
                faces[3*i+j] = vertices[j];
            }

            // Extract all edges from this face
            for (int j = 0; j < vertex_count_face; ++j) {
                if (_ei >= (3*face_count)){
                    printf("Broke early! Error! _ei: %d, limit: %d\n", _ei, 3*face_count);
                    break;
                }
                int a = vertices[j];
                int b = vertices[(j + 1) % vertex_count_face];
                e0[_ei] = std::min(a, b);
                e1[_ei] = std::max(a, b);
                _ei += 1;
            }
        }

    } else {
        // ASCII format reading
        for (int i = 0; i < vertex_count; ++i) {
            // -- read --
            float x, y, z;
            int r, g, b, alpha;
            file >> x >> y >> z >> r >> g >> b >> alpha;
            
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
            // pos[3*i+0] = x;
            // pos[3*i+1] = y;
            // pos[3*i+2] = z;
            // ftr[3*i+0] = r / 255.0f;
            // ftr[3*i+1] = g / 255.0f;
            // ftr[3*i+2] = b / 255.0f;

            // -- append --
            float3 pos_i = make_float3(x,y,z);
            pos[i] = pos_i;
            float3 ftr_i = make_float3(r/255.0f,g/255.0f,b/255.0f);
            ftr[i] = ftr_i;
        }
        
        // Read faces for ASCII format (you'll need to implement this if needed)
        // This was missing in the original ASCII branch
    }
    
    size = vertex_count;
    nfaces = face_count;
    file.close();
    return true;
};

bool ScanNetScene::write_ply_with_fn(const std::filesystem::path& scene_path, 
                            const std::filesystem::path& output_root, PointCloudDataHost& data){
                            // float3* ftrs, float3* pos, uint32_t* edges, 
                            // int nnodes, int nedges, uint8_t* gcolors, uint32_t* labels) {

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
    //return write_ply(ply_file,ftrs,pos,edges,nnodes,nedges,gcolors,labels);
    return write_ply(ply_file,data);
}


bool ScanNetScene::write_ply(const std::filesystem::path& ply_file, PointCloudDataHost& data) {

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
    header += "element vertex " + std::to_string(data.V) + "\n";
    header += "property float x\n";
    header += "property float y\n";
    header += "property float z\n";
    header += "property uchar red\n";
    header += "property uchar green\n";
    header += "property uchar blue\n";
    header += "property uchar alpha\n";
    if (!data.gcolors.empty()){
        header += "property uchar gcolor\n";
    }
    if (!data.labels.empty()){
        header += "property uint label\n";
    }
    header += "element edge " + std::to_string(data.E) + '\n';
    header += "property int vertex1\n";
    header += "property int vertex2\n";
    header += "end_header\n";
    file.write(header.c_str(), header.length());

    // -- write data --
    for (int i = 0; i < data.V; ++i) {
        
        // -- init --
        float x, y, z;
        unsigned char r, g, b, alpha;
        uint32_t label;
        uint8_t gcolor_id;

        // -- unpack --
        float3 ftr_i = data.ftrs[i];
        float3 pos_i = data.pos[i];
        x = pos_i.x;
        y = pos_i.y;
        z = pos_i.z;
        r = static_cast<unsigned char>(ftr_i.x * 255.0f);
        g = static_cast<unsigned char>(ftr_i.y * 255.0f);
        b = static_cast<unsigned char>(ftr_i.z * 255.0f);
        alpha = 255;
        gcolor_id = (!data.gcolors.empty()) ? data.gcolors[i] : 0;
        label =  (!data.labels.empty())? data.labels[i] : 0;
        //printf("label: %ld\n",label);
        // if (i < 10){
        //     printf("x,y,z r,g,b: %2.2f %2.2f %2.2f %d %d %d\n",x,y,z,r,g,b);
        // }

        // -- write --
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));
        // -- write gcolor if provided --
        if (!data.gcolors.empty()) {
            file.write(reinterpret_cast<const char*>(&gcolor_id), sizeof(unsigned char));
        }
        // -- write label if provided --
        if (!data.labels.empty()) {
            file.write(reinterpret_cast<const char*>(&label), sizeof(uint32_t));
        }

    }

    for (int i = 0; i < data.E; ++i){
        //unsigned char len = 2;
        int e0 = data.edges[2*i+0]; // ??
        int e1 = data.edges[2*i+1];
        // if ((i % 10000 == 0) || (i < 10) || (i > (data.E-10))) {        
        //     printf("e0, e1: %d %d\n",e0,e1);
        // }
        //file.write(reinterpret_cast<const char*>(&len), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&e0), sizeof(int));
        file.write(reinterpret_cast<const char*>(&e1), sizeof(int));
    }
    
    file.close();
    return true;

};


// -- write each scene; [nnodes == spix if point-cloud is the superpixel point cloud] --
bool write_spix(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, SuperpixelParams3d& spix_params){
   
    // -- sync before io --
    cudaDeviceSynchronize();
    thrust::host_vector<uint32_t> csum_nspix = spix_params.csum_nspix;

    int bx = 0;
    for (const auto& scene_file : scene_files) {

        // -- get batch slice --
        int start_idx = csum_nspix[bx];
        int end_idx = csum_nspix[bx + 1];
        int nspix = end_idx - start_idx;
        thrust::host_vector<float3> mu_app(spix_params.mu_app.begin() + start_idx,
                                        spix_params.mu_app.begin() + end_idx);
        thrust::host_vector<double3> mu_pos(spix_params.mu_pos.begin() + start_idx,
                                            spix_params.mu_pos.begin() + end_idx);
        thrust::host_vector<double3> var_pos(spix_params.var_pos.begin() + start_idx,
                                            spix_params.var_pos.begin() + end_idx);
        thrust::host_vector<double3> cov_pos(spix_params.cov_pos.begin() + start_idx,
                                            spix_params.cov_pos.begin() + end_idx);

        // -- write --
        ScanNetScene scene;
        // if(!scene.write_spix_ply_with_fn(scene_file,output_root,mu_app,mu_pos,var_pos,cov_pos,nspix)){
        //     exit(1);
        // }

        bx += 1;
    }

    return 0;
}


// bool ScanNetScene::write_spix_ply_with_fn(const std::filesystem::path& scene_path, 
//                                   const std::filesystem::path& output_root,
//                                   thrust::host_vector<float3>& ftrs, 
//                                   thrust::host_vector<double3>& pos,
//                                   thrust::host_vector<double3>& var, 
//                                   thrust::host_vector<double3>& cov, int nspix) {

//     // -- helper --
//     std::string line;

//     // -- make dir if needed --
//     std::string scene_name = scene_path.filename().string();
//     std::filesystem::path write_path = output_root / scene_name;
//     if (!std::filesystem::exists(write_path)) {
//         std::filesystem::create_directories(write_path);
//     }
//     // -- get filenames --
//     std::filesystem::path ply_file = write_path / (scene_name + "_spix.ply");
//     std::cout << ply_file << std::endl;
    
//     // thrust::host_vector<uint32_t> border_edges(0); // spoof for now.
//     // thrust::host_vector<uint32_t> border_ptr(0); // spoof for now.
//     return this->write_spix_ply(ply_file,ftrs,pos,var,cov,nspix);

// }


// bool ScanNetScene::write_spix_ply(const std::filesystem::path& ply_file,
//                                   thrust::host_vector<float3>& ftrs, 
//                                   thrust::host_vector<double3>& pos,
//                                   thrust::host_vector<double3>& var, 
//                                   thrust::host_vector<double3>& cov, 
//                                 //   thrust::host_vector<uint32_t>& border_edges,
//                                 //   thrust::host_vector<uint32_t>& border_ptr,
//                                   int nspix) {
    
//     // -- delete existing file if it exists --
//     if (std::filesystem::exists(ply_file)) {
//         std::filesystem::remove(ply_file);
//     }

//     // -- open file for writing --
//     std::ofstream file(ply_file.string(), std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Can not open: " << ply_file.string() << std::endl;
//         return false;
//     }

//     // -- write PLY header --
//     std::string header = "ply\n";
//     header += "format binary_little_endian 1.0\n";
//     header += "comment MLIB generated\n";
//     header += "element vertex " + std::to_string(nspix) + "\n";
//     header += "property float x\n";
//     header += "property float y\n";
//     header += "property float z\n";
//     header += "property float var_x\n";
//     header += "property float var_y\n";
//     header += "property float var_z\n";
//     header += "property float cov_xy\n";
//     header += "property float cov_xz\n";
//     header += "property float cov_yz\n";
//     header += "property uchar red\n";
//     header += "property uchar green\n";
//     header += "property uchar blue\n";
//     header += "property uchar alpha\n";
//     // if (border_edges.size() > 0){
//     //     header += "element edge " + std::to_string(border_edges.size()/2) + '\n';
//     //     header += "property int vertex1\n";
//     //     header += "property int vertex2\n";
//     //     // header += "element face " + std::to_string(border_edges.size()/2) + "\n";
//     //     // header += "property list uchar int vertex_indices\n";
//     // }
//     header += "end_header\n";
//     file.write(header.c_str(), header.length());

//     // -- write data --
//     for (int i = 0; i < nspix; ++i) {
        
//         // -- init --
//         unsigned char r, g, b, alpha;
//         uint32_t label;
//         uint8_t gcolor_id;

//         // -- unpack --
//         float3 _ftrs = ftrs[i];
//         double3 _pos = pos[i];
//         double3 _var = var[i];
//         double3 _cov = cov[i];
//         float x = _pos.x;
//         float y = _pos.y;
//         float z = _pos.z;
//         float var_x = _var.x;
//         float var_y = _var.y;
//         float var_z = _var.z;
//         float cov_x = _cov.x;
//         float cov_y = _cov.y;
//         float cov_z = _cov.z;

//         r = static_cast<unsigned char>(_ftrs.x * 255.0f);
//         g = static_cast<unsigned char>(_ftrs.y * 255.0f);
//         b = static_cast<unsigned char>(_ftrs.z * 255.0f);
//         alpha = 255;
//         //printf("label: %ld\n",label);
//         //printf("x,y,z r,g,b: %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n",x,y,z,ftrs[3*i+0],ftrs[3*i+1] ,ftrs[3*i+2] );


//         // -- write --
//         file.write(reinterpret_cast<const char*>(&x), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&y), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&z), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&var_x), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&var_y), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&var_z), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&cov_x), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&cov_y), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&cov_z), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));
//     }


//     file.close();
//     return true;

// };




// PointCloudData read_scene(const std::vector<std::filesystem::path>& scene_files){

//     // -- read sizes --
//     int batchsize = scene_files.size();

//     // -- batch offset --
//     std::vector<int> ptr_host(batchsize+1,0);
//     std::vector<int> eptr_host(batchsize+1,0);
//     std::vector<int> face_ptr_host(batchsize+1,0);
    
//     // First pass: get sizes
//     int _ix = 1;
//     for (const auto& scene_file : scene_files) {
//         ptr_host[_ix] = get_vertex_count(scene_file)+ptr_host[_ix-1];
//         face_ptr_host[_ix] = get_face_count(scene_file)+face_ptr_host[_ix-1];
//         _ix++;
//     }
//     int total_nodes = ptr_host[batchsize];
//     int total_faces = face_ptr_host[batchsize];
//     printf("total nodes, total faces: %d, %d\n",total_nodes,total_faces);

//     // Reserve space for all points (host vectors)
//     std::vector<float3> ftrs_host(total_nodes);
//     std::vector<float3> pos_host(total_nodes);
//     std::vector<float> dim_sizes_host(6*batchsize,0);
//     std::vector<uint8_t> bids_host(total_nodes,0);
//     std::vector<uint32_t> faces_host(3*total_faces,0);
//     thrust::device_vector<uint32_t> edges{};
//     std::vector<uint8_t> ebids_host{};

//     // Second pass: append data
//     int _bx = 0;
//     _ix = 0;
//     for (const auto& scene_file : scene_files) {

//         // -- .. --
//         ScanNetScene scene;
//         if(!scene.read_ply(scene_file)){
//             exit(1);
//         }

//         // -- .. --
//         memcpy(&ftrs[3*_ix], scene.ftr.data(), 3 * scene.size * sizeof(float));
//         memcpy(&pos[3*_ix], scene.pos.data(), 3 * scene.size * sizeof(float));
//         std::fill(bids.begin() + _ix, bids.begin() + _ix + scene.size, _bx);

//         int nfaces_batch = face_ptr_cpu[_bx+1] - face_ptr_cpu[_bx];
//         memcpy(&faces[3*face_ptr_cpu[_bx]], scene.faces.data(), 3 * nfaces_batch * sizeof(uint32_t));

//         // -- extract edges from edge-pairs --
//         thrust::device_vector<uint32_t> edges_b = extract_edges_from_pairs(scene.e0,scene.e1);
//         size_t _size = edges.size();
//         // edges.resize(_size + edges_b.size());
//         // cudaDeviceSynchronize();
//         // thrust::copy(edges_b.begin(), edges_b.end(), edges.begin() + _size);
//         edges.insert(edges.end(), edges_b.begin(), edges_b.end());
//         //ebids.resize(_size/2 + edges_b.size()/2, _bx);
//         ebids.insert(ebids.end(), edges_b.size()/2, _bx);
//         eptr_cpu[_bx+1] = eptr_cpu[_bx]+edges_b.size()/2;
//         printf("eptr_cpu: %d %d\n", eptr_cpu[_bx],eptr_cpu[_bx+1]);

//         // -- .. --
//         dim_sizes[6*_bx+0] = scene.xmin;
//         dim_sizes[6*_bx+1] = scene.xmax;
//         dim_sizes[6*_bx+2] = scene.ymin;
//         dim_sizes[6*_bx+3] = scene.ymax;
//         dim_sizes[6*_bx+4] = scene.zmin;
//         dim_sizes[6*_bx+5] = scene.zmax;
//         _ix += scene.size;
//         _bx += 1;

//         // // -- .. --
//         // ScanNetScene scene;
//         // if(!scene.read_ply(scene_file)){
//         //     exit(1);
//         // }

//         // // -- copy features and positions --
//         // for(int i = 0; i < scene.size; i++) {
//         //     ftrs_host[_ix + i] = make_float3(scene.ftr[3*i], scene.ftr[3*i+1], scene.ftr[3*i+2]);
//         //     pos_host[_ix + i] = make_float3(scene.pos[3*i], scene.pos[3*i+1], scene.pos[3*i+2]);
//         // }
        
//         // // -- set batch ids --
//         // std::fill(bids_host.begin() + _ix, bids_host.begin() + _ix + scene.size, _bx);

//         // // -- copy faces --
//         // int nfaces_batch = face_ptr_host[_bx+1] - face_ptr_host[_bx];
//         // memcpy(&faces_host[3*face_ptr_host[_bx]], scene.faces.data(), 3 * nfaces_batch * sizeof(uint32_t));

//         // // -- extract edges from edge-pairs --
//         // thrust::device_vector<uint32_t> edges_b = extract_edges_from_pairs(scene.e0,scene.e1);
//         // edges.insert(edges.end(), edges_b.begin(), edges_b.end());
//         // ebids_host.insert(ebids_host.end(), edges_b.size()/2, _bx);
//         // eptr_host[_bx+1] = eptr_host[_bx]+edges_b.size()/2;
//         // printf("eptr_host: %d %d\n", eptr_host[_bx],eptr_host[_bx+1]);

//         // // -- copy dimension sizes --
//         // dim_sizes_host[6*_bx+0] = scene.xmin;
//         // dim_sizes_host[6*_bx+1] = scene.xmax;
//         // dim_sizes_host[6*_bx+2] = scene.ymin;
//         // dim_sizes_host[6*_bx+3] = scene.ymax;
//         // dim_sizes_host[6*_bx+4] = scene.zmin;
//         // dim_sizes_host[6*_bx+5] = scene.zmax;
//         // _ix += scene.size;
//         // _bx += 1;
//     }

//     // -- view --
//     printf("_ix: %d\n",_ix);
//     std::cout << "=== Scene Batch Info ===" << std::endl;
//     std::cout << "Batch size: " << batchsize << std::endl;
//     std::cout << "Total nodes: " << total_nodes << std::endl;

//     // Print per-scene info
//     for (int i = 0; i < batchsize; ++i) {
//         std::cout << "Scene " << i << ": " << (ptr_host[i+1] - ptr_host[i]) << " points" << std::endl;
//         std::cout << "  Bounds: [" << dim_sizes_host[6*i+0] << ", " << dim_sizes_host[6*i+1] << "] "
//                 << "[" << dim_sizes_host[6*i+2] << ", " << dim_sizes_host[6*i+3] << "] "
//                 << "[" << dim_sizes_host[6*i+4] << ", " << dim_sizes_host[6*i+5] << "]" << std::endl;
//     }

//     // Print first few points
//     std::cout << "First 3 points:" << std::endl;
//     for (int i = 0; i < std::min(3, total_nodes); i++) {
//         std::cout << "  Pos: (" << pos_host[i].x << ", " << pos_host[i].y << ", " << pos_host[i].z << ")" << std::endl;
//         std::cout << "  RGB: (" << ftrs_host[i].x << ", " << ftrs_host[i].y << ", " << ftrs_host[i].z << ")" << std::endl;
//     }

//     // Create device vectors and copy data
//     int nedges = eptr_host[batchsize];
//     printf("nedges: %d,%d\n",nedges,ebids_host.size());
    
//     thrust::device_vector<uint8_t> gcolors_host;
//     thrust::device_vector<uint32_t> csr_edges_host;
//     thrust::device_vector<uint32_t> csr_eptr_host;


//     PointCloudData result(
//         ftrs_host, pos_host, faces_host, edges, 
//         bids_host, ptr_host, eptr_host, fptr_host,
//         dim_sizes_host, 0, batchsize, total_nodes, nedges, total_faces
//     );

//     return result;

// }



    // Create device vectors by moving/copying from host data
    // thrust::device_vector<float3> ftrs_device(ftrs_host.begin(), ftrs_host.end());
    // thrust::device_vector<float3> pos_device(pos_host.begin(), pos_host.end());
    // thrust::device_vector<uint32_t> faces_device(faces_host.begin(), faces_host.end());
    // thrust::device_vector<uint8_t> gcolors_device; // Empty for now - you might want to populate this
    // thrust::device_vector<uint32_t> csr_edges_device(std::move(edges)); // Move the edges we built
    // thrust::device_vector<uint32_t> csr_eptr_device; // You might want to populate this based on your CSR needs
    // thrust::device_vector<uint8_t> bids_device(bids_host.begin(), bids_host.end());
    // thrust::device_vector<int> ptr_device(ptr_host, ptr_host + batchsize + 1);
    // thrust::device_vector<int> eptr_device(eptr_host, eptr_host + batchsize + 1);
    // thrust::device_vector<int> fptr_device(face_ptr_host, face_ptr_host + batchsize + 1);
    // // thrust::device_vector<float> dim_sizes_device(dim_sizes_host.begin(), dim_sizes_host.end());

    // // Calculate edge count for the struct
    // int E = csr_edges_device.size() / 2; // assuming edges are stored as pairs
    
    // // // Create and return PointCloudData struct
    // return PointCloudData(
    //     std::move(ftrs_device),
    //     std::move(pos_device), 
    //     std::move(faces_device),
    //     std::move(gcolors_device),
    //     std::move(csr_edges_device),
    //     std::move(csr_eptr_device),
    //     std::move(bids_device),
    //     std::move(ptr_device),
    //     std::move(eptr_device),
    //     std::move(fptr_device),
    //     std::move(dim_sizes_device),
    //     0, // gchrome - set to appropriate value
    //     batchsize, // B
    //     total_nodes, // V
    //     E, // E
    //     total_faces // F
    // );


// -- read each scene --
// std::tuple<float3*,float3*,uint32_t*,uint8_t*,uint8_t*,int*,int*,float*,uint32_t*,int*>
// read_scene(const std::vector<std::filesystem::path>& scene_files){

//     // -- read sizes --
//     int batchsize = scene_files.size();
//     int* ptr_cpu = (int*)malloc((batchsize+1) * sizeof(int));
//     int* face_ptr_cpu = (int*)malloc((batchsize+1) * sizeof(int));
    
//     // First pass: get sizes
//     int _ix = 1;
//     ptr_cpu[0] = 0;
//     face_ptr_cpu[0] = 0;
//     for (const auto& scene_file : scene_files) {
//         ptr_cpu[_ix] = get_vertex_count(scene_file)+ptr_cpu[_ix-1];
//         face_ptr_cpu[_ix] = get_face_count(scene_file)+face_ptr_cpu[_ix-1];
//         _ix++;
//     }
//     int total_nodes = ptr_cpu[batchsize];
//     int total_faces = face_ptr_cpu[batchsize];
//     printf("total nodes, total faces: %d, %d\n",total_nodes,total_faces);

//     // Reserve space for all points
//     std::vector<float> ftrs(3*total_nodes,0);
//     std::vector<float> pos(3*total_nodes,0);
//     std::vector<float> dim_sizes(6*batchsize,0);
//     std::vector<uint8_t> bids(total_nodes,0);
//     std::vector<uint32_t> faces(3*total_faces,0);
//     thrust::device_vector<uint32_t> edges{};
//     std::vector<uint8_t> ebids{};
//     int* eptr_cpu = (int*)malloc((batchsize+1) * sizeof(int));
//     eptr_cpu[0] = 0;

//     // Second pass: append data
//     int _bx = 0;
//     _ix = 0;
//     for (const auto& scene_file : scene_files) {

//         // -- .. --
//         ScanNetScene scene;
//         if(!scene.read_ply(scene_file)){
//             exit(1);
//         }

//         // -- .. --
//         memcpy(&ftrs[3*_ix], scene.ftr.data(), 3 * scene.size * sizeof(float));
//         memcpy(&pos[3*_ix], scene.pos.data(), 3 * scene.size * sizeof(float));
//         std::fill(bids.begin() + _ix, bids.begin() + _ix + scene.size, _bx);

//         int nfaces_batch = face_ptr_cpu[_bx+1] - face_ptr_cpu[_bx];
//         memcpy(&faces[3*face_ptr_cpu[_bx]], scene.faces.data(), 3 * nfaces_batch * sizeof(uint32_t));

//         // -- extract edges from edge-pairs --
//         thrust::device_vector<uint32_t> edges_b = extract_edges_from_pairs(scene.e0,scene.e1);
//         size_t _size = edges.size();
//         // edges.resize(_size + edges_b.size());
//         // cudaDeviceSynchronize();
//         // thrust::copy(edges_b.begin(), edges_b.end(), edges.begin() + _size);
//         edges.insert(edges.end(), edges_b.begin(), edges_b.end());
//         //ebids.resize(_size/2 + edges_b.size()/2, _bx);
//         ebids.insert(ebids.end(), edges_b.size()/2, _bx);
//         eptr_cpu[_bx+1] = eptr_cpu[_bx]+edges_b.size()/2;
//         printf("eptr_cpu: %d %d\n", eptr_cpu[_bx],eptr_cpu[_bx+1]);

//         // -- .. --
//         dim_sizes[6*_bx+0] = scene.xmin;
//         dim_sizes[6*_bx+1] = scene.xmax;
//         dim_sizes[6*_bx+2] = scene.ymin;
//         dim_sizes[6*_bx+3] = scene.ymax;
//         dim_sizes[6*_bx+4] = scene.zmin;
//         dim_sizes[6*_bx+5] = scene.zmax;
//         _ix += scene.size;
//         _bx += 1;
//     }

//     // -- view --
//     printf("_ix: %d\n",_ix);
//     std::cout << "=== Scene Batch Info ===" << std::endl;
//     std::cout << "Batch size: " << batchsize << std::endl;
//     std::cout << "Total nodes: " << total_nodes << std::endl;

//     // Print per-scene info
//     int offset = 0;
//     for (int i = 0; i < batchsize; ++i) {
//         std::cout << "Scene " << i << ": " << (ptr_cpu[i+1] - ptr_cpu[i]) << " points" << std::endl;
//         std::cout << "  Bounds: [" << dim_sizes[6*i+0] << ", " << dim_sizes[6*i+1] << "] "
//                 << "[" << dim_sizes[6*i+2] << ", " << dim_sizes[6*i+3] << "] "
//                 << "[" << dim_sizes[6*i+4] << ", " << dim_sizes[6*i+5] << "]" << std::endl;
//     }

//     // Print first few points
//     std::cout << "First 3 points:" << std::endl;
//     for (int i = 0; i < std::min(9, (int)pos.size()); i += 3) {
//         std::cout << "  Pos: (" << pos[i] << ", " << pos[i+1] << ", " << pos[i+2] << ")" << std::endl;
//         std::cout << "  RGB: (" << ftrs[i] << ", " << ftrs[i+1] << ", " << ftrs[i+2] << ")" << std::endl;
//     }

//     // -- copy --
//     int nedges = eptr_cpu[batchsize];
//     printf("nedges: %d,%d\n",nedges,ebids.size());
//     float3* ftrs_cu = (float3*)easy_allocate(total_nodes,sizeof(float3));
//     float3* pos_cu = (float3*)easy_allocate(total_nodes,sizeof(float3));
//     uint32_t* edges_cu = (uint32_t*)easy_allocate(2*nedges,sizeof(uint32_t));
//     uint8_t* bids_cu = (uint8_t*)easy_allocate(total_nodes,sizeof(uint8_t));
//     uint8_t* ebids_cu = (uint8_t*)easy_allocate(nedges,sizeof(uint8_t));
//     int* ptr_cu = (int*)easy_allocate(batchsize+1,sizeof(int));
//     int* eptr_cu = (int*)easy_allocate(batchsize+1,sizeof(int));
//     float* dim_sizes_cu = (float*)easy_allocate(6*batchsize,sizeof(float));

//     uint32_t* faces_cu = (uint32_t*)easy_allocate(total_faces,sizeof(uint32_t));
//     int* face_ptr_cu = (int*)easy_allocate(batchsize+1,sizeof(int));


//     cudaDeviceSynchronize();
//     cudaMemcpy(ftrs_cu,thrust::raw_pointer_cast(ftrs.data()),total_nodes*sizeof(float3),cudaMemcpyHostToDevice);
//     cudaMemcpy(pos_cu,thrust::raw_pointer_cast(pos.data()),total_nodes*sizeof(float3),cudaMemcpyHostToDevice);
//     cudaMemcpy(edges_cu,thrust::raw_pointer_cast(edges.data()),2*nedges*sizeof(uint32_t),cudaMemcpyDeviceToDevice);
//     cudaMemcpy(bids_cu,thrust::raw_pointer_cast(bids.data()),total_nodes*sizeof(uint8_t),cudaMemcpyHostToDevice);
//     cudaMemcpy(ebids_cu,thrust::raw_pointer_cast(ebids.data()),nedges*sizeof(uint8_t),cudaMemcpyHostToDevice);
//     cudaMemcpy(ptr_cu,ptr_cpu,(batchsize+1)*sizeof(int),cudaMemcpyHostToDevice);
//     cudaMemcpy(eptr_cu,eptr_cpu,(batchsize+1)*sizeof(int),cudaMemcpyHostToDevice);
//     cudaMemcpy(dim_sizes_cu,thrust::raw_pointer_cast(dim_sizes.data()),6*batchsize*sizeof(float),cudaMemcpyHostToDevice);

//     cudaMemcpy(faces_cu,thrust::raw_pointer_cast(faces.data()),total_faces*sizeof(uint32_t),cudaMemcpyHostToDevice);
//     cudaMemcpy(face_ptr_cu,face_ptr_cpu,(batchsize+1)*sizeof(int),cudaMemcpyHostToDevice);

//     // -- free --
//     free(ptr_cpu);
//     free(eptr_cpu);

//     return std::tuple(ftrs_cu,pos_cu,edges_cu,bids_cu,ebids_cu,ptr_cu,eptr_cu,dim_sizes_cu,faces_cu,face_ptr_cu);

// }


// if (border_edges.size()>0){
//     // uint32_t nedges = border_edges.size()/2;
//     // for (uint32_t i = 0; i < nedges; ++i) {
//     //     uint32_t start = border_ptr[i];
//     //     uint32_t end = border_ptr[i+1];
//     //     unsigned char spix_size = end - start;
//     //     file.write(reinterpret_cast<char*>(&spix_size), sizeof(unsigned char));
//     //     for (uint32_t index=start; index < end; index++){
//     //         uint32_t vertex_id = border_edges[index];
//     //         file.write(reinterpret_cast<char*>(&vertex_id), sizeof(int));
//     //         printf("[%d,%d] %d\n",index,spix_size,vertex_id);
//     //     }
//     // }
//     uint32_t nedges = border_edges.size()/2;
//     for (uint32_t i = 0; i < nedges; ++i){
//         uint32_t e0 = border_edges[2*i+0]; // ??
//         uint32_t e1 = border_edges[2*i+1];
//         if ((i % 10000 == 0) || (i < 10) || (i > (nedges-10))) {        
//             printf("e0, e1: %d %d\n",e0,e1);
//         }
//         //file.write(reinterpret_cast<const char*>(&len), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&e0), sizeof(int));
//         file.write(reinterpret_cast<const char*>(&e1), sizeof(int));
//     }
    
// }


// bool write_ply_csr_edges(const std::filesystem::path& ply_file,
//                         float* ftrs, float* pos, uint32_t* csr_edges,  uint32_t* csr_eptr,  
//                         int V, int E){

//     // -- delete existing file if it exists --
//     if (std::filesystem::exists(ply_file)) {
//         std::filesystem::remove(ply_file);
//     }
//     bool* gcolors = nullptr;
//     uint32_t* labels = nullptr;

//     // -- open file for writing --
//     std::ofstream file(ply_file.string(), std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Can not open: " << ply_file.string() << std::endl;
//         return false;
//     }

//     // -- write PLY header --
//     std::string header = "ply\n";
//     header += "format binary_little_endian 1.0\n";
//     header += "comment MLIB generated\n";
//     header += "element vertex " + std::to_string(V) + "\n";
//     header += "property float x\n";
//     header += "property float y\n";
//     header += "property float z\n";
//     header += "property uchar red\n";
//     header += "property uchar green\n";
//     header += "property uchar blue\n";
//     header += "property uchar alpha\n";
//     if (gcolors == nullptr){
//         header += "property uchar gcolor\n";
//     }
//     if (labels != nullptr) {
//         header += "property uint label\n";
//     }
//     header += "element edge " + std::to_string(E) + '\n';
//     header += "property int vertex1\n";
//     header += "property int vertex2\n";
//     header += "end_header\n";
//     file.write(header.c_str(), header.length());

//     // -- write data --
//     for (int i = 0; i < V; ++i) {
        
//         // -- init --
//         float x, y, z;
//         unsigned char r, g, b, alpha;
//         uint32_t label;
//         uint8_t gcolor_id;

//         // -- unpack --
//         x = pos[3*i+0];
//         y = pos[3*i+1];
//         z = pos[3*i+2];
//         r = static_cast<unsigned char>(ftrs[3*i+0] * 255.0f);
//         g = static_cast<unsigned char>(ftrs[3*i+1] * 255.0f);
//         b = static_cast<unsigned char>(ftrs[3*i+2] * 255.0f);
//         alpha = 255;
//         gcolor_id = (gcolors != nullptr) ? gcolors[i] : 0;
//         label = (labels != nullptr) ? static_cast<uint32_t>(labels[i]) : 0;
//         //printf("label: %ld\n",label);
//         //printf("x,y,z r,g,b: %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n",x,y,z,ftrs[3*i+0],ftrs[3*i+1] ,ftrs[3*i+2] );


//         // -- write --
//         file.write(reinterpret_cast<const char*>(&x), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&y), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&z), sizeof(float));
//         file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
//         file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));
//         // -- write gcolor if provided --
//         if (gcolors != nullptr) {
//             file.write(reinterpret_cast<const char*>(&gcolor_id), sizeof(unsigned char));
//         }
//         // -- write label if provided --
//         if (labels != nullptr) {
//             file.write(reinterpret_cast<const char*>(&label), sizeof(uint32_t));
//         }

//     }

//     for (int i = 0; i < E; ++i){
//         //unsigned char len = 2;
//         int e0 = edges[2*i+0]; // ??
//         int e1 = edges[2*i+1];

//         for (int jx = start; jx < end; jx++){
//             file.write(reinterpret_cast<const char*>(&e0), sizeof(int));
//             file.write(reinterpret_cast<const char*>(&e1), sizeof(int));
//         }
//     }
    
//     file.close();
//     return true;                   

// }



// bool ScanNetScene::read_ply(const std::filesystem::path& scene_path) {

//     // -- helper --
//     std::string line;

//     // -- get filenames --
//     std::string scene_name = scene_path.filename().string();
//     std::filesystem::path info_file = scene_path / (scene_name + ".txt");
//     std::filesystem::path ply_file = scene_path / (scene_name + "_vh_clean_2.ply");
//     std::cout << info_file << std::endl;
//     std::cout << ply_file << std::endl;

//     // -- read the axis alignment matrix --
//     float axis_align[16];
//     for (int index = 0; index < 16; index++) {
//         int i = index / 4;
//         int j = index % 4;
//         axis_align[index] = 1.0 * (i==j);
//     }
//     if (std::filesystem::exists(info_file)){
//         std::ifstream info_stream(info_file);
//         while (std::getline(info_stream, line)) {
//             if (line.rfind("axisAlignment", 0) == 0) { // starts with "axisAlignment"
//                 std::istringstream iss(line.substr(line.find('=') + 1));
//                 for (int i = 0; i < 16; i++) {
//                     if (!(iss >> axis_align[i])) {
//                         std::cerr << "Error parsing axisAlignment\n";
//                         return 1;
//                     }
//                 }
//                 break;
//             }
//         }
//         info_stream.close();
//     }

//     // // Print result to verify
//     // for (int i = 0; i < 16; i++) {
//     //     std::cout << axis_align[i] << (i % 4 == 3 ? "\n" : " ");
//     // }

//     // -- read ply file --
//     std::cout << ply_file.string() << std::endl;
//     std::ifstream file(ply_file.string(), std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Can not open: " << ply_file << std::endl;
//         return false;
//     }
    
//     // Parse header
//     int vertex_count = 0;
//     int face_count = 0;
//     bool binary_format = false;
    
//     while (std::getline(file, line)) {
//         std::cout << line << std::endl;
//         if (line.find("element vertex") != std::string::npos) {
//             vertex_count = std::stoi(line.substr(15));
//         } else if (line.find("element face") != std::string::npos) {
//             face_count = std::stoi(line.substr(13));
//         } else if (line.find("format binary") != std::string::npos) {
//             binary_format = true;
//         } else if (line == "end_header") {
//             break;
//         }
//     }
//     printf("vertex count: %d\n",vertex_count);
//     printf("face count: %d\n",face_count);


//     // Allocate vectors
//     pos.reserve(3*vertex_count);
//     ftr.reserve(3*vertex_count);
//     e0.resize(3*face_count); // triangles
//     e1.resize(3*face_count);
//     faces.resize(3*face_count);

    
//     // Read data
//     int _ei = 0;
//     if (binary_format) {
//         for (int i = 0; i < vertex_count; ++i) {
//             float x, y, z;
//             unsigned char r, g, b, alpha;
            
//             // -- read --
//             file.read(reinterpret_cast<char*>(&x), sizeof(float));
//             file.read(reinterpret_cast<char*>(&y), sizeof(float));
//             file.read(reinterpret_cast<char*>(&z), sizeof(float));
//             file.read(reinterpret_cast<char*>(&r), sizeof(unsigned char));
//             file.read(reinterpret_cast<char*>(&g), sizeof(unsigned char));
//             file.read(reinterpret_cast<char*>(&b), sizeof(unsigned char));
//             file.read(reinterpret_cast<char*>(&alpha), sizeof(unsigned char));

//             // -- axis align --
//             float x_new = axis_align[0]*x + axis_align[1]*y + axis_align[2]*z + axis_align[3];
//             float y_new = axis_align[4]*x + axis_align[5]*y + axis_align[6]*z + axis_align[7];
//             float z_new = axis_align[8]*x + axis_align[9]*y + axis_align[10]*z + axis_align[11];

//             // -- update --
//             x = x_new;
//             y = y_new;
//             z = z_new;

//             // -- update bounds --
//             xmin = std::min(xmin, x);
//             xmax = std::max(xmax, x);
//             ymin = std::min(ymin, y);
//             ymax = std::max(ymax, y);
//             zmin = std::min(zmin, z);
//             zmax = std::max(zmax, z);
            
//             // -- append --
//             pos[3*i+0] = x;
//             pos[3*i+1] = y;
//             pos[3*i+2] = z;
//             ftr[3*i+0] = r / 255.0f;
//             ftr[3*i+1] = g / 255.0f;
//             ftr[3*i+2] = b / 255.0f;
//             // e0[_ei] = i;
//             // e1[_ei] = i;
//             // _ei++;
//         }

//         // -- read edges --
//         for (int i = 0; i < face_count; ++i) {
//             unsigned char vertex_count;
//             file.read(reinterpret_cast<char*>(&vertex_count), sizeof(unsigned char));
//             if (vertex_count != 3){
//                 printf("vertex count: %d\n",vertex_count);
//             }
//             std::vector<int> vertices(vertex_count);
//             for (int j = 0; j < vertex_count; ++j) {
//                 file.read(reinterpret_cast<char*>(&vertices[j]), sizeof(int));
//                 faces[3*i+j] = vertices[j];
//             }

//             // Extract all edges from this face
//             for (int j = 0; j < vertex_count; ++j) {
//                 if ((_ei-vertex_count) >= (6*face_count)){
//                     printf("Broke early! Error!\n");
//                     exit(1);
//                 }
//                 int a = vertices[j];
//                 int b = vertices[(j + 1) % vertex_count];
//                 // if (a == b){ 
//                 //     printf("self loop? %d\n",a);
//                 //     continue; 
//                 // }
//                 e0[_ei] = std::min(a, b);
//                 e1[_ei] = std::max(a, b);
//                 _ei += 1;
//             }
//         }
//         //printf("n pairs: %d\n",_ei);
//         assert(_ei == e0.size()); // must fill the vector.

//     } else {
//         for (int i = 0; i < vertex_count; ++i) {
//             // -- read --
//             float x, y, z;
//             int r, g, b, alpha;
//             file >> x >> y >> z >> r >> g >> b >> alpha;
//             //printf("x,y,z: %.2f %.2f %.2f\n",x,y,z);
            
//             // -- axis align --
//             float x_new = axis_align[0]*x + axis_align[1]*y + axis_align[2]*z + axis_align[3];
//             float y_new = axis_align[4]*x + axis_align[5]*y + axis_align[6]*z + axis_align[7];
//             float z_new = axis_align[8]*x + axis_align[9]*y + axis_align[10]*z + axis_align[11];

//             // -- update --
//             x = x_new;
//             y = y_new;
//             z = z_new;

//             // -- update bounds --
//             xmin = std::min(xmin, x);
//             xmax = std::max(xmax, x);
//             ymin = std::min(ymin, y);
//             ymax = std::max(ymax, y);
//             zmin = std::min(zmin, z);
//             zmax = std::max(zmax, z);

//             // -- append --
//             pos[3*i+0] = x;
//             pos[3*i+1] = y;
//             pos[3*i+2] = z;
//             ftr[3*i+0] = r / 255.0f;
//             ftr[3*i+1] = g / 255.0f;
//             ftr[3*i+2] = b / 255.0f;
//         }
//     }
    
//     size = vertex_count;
//     nfaces = face_count;
//     //nfaces = 3*face_count+vertex_count;
//     file.close();
//     return true;
// };
