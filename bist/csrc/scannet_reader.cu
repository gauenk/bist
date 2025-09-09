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
    std::vector<uint32_t> faces_eptr(total_faces+1, 0);
    std::vector<uint8_t> face_batch_ids(total_faces, 0);

    
    // Edge data (stored on device)
    thrust::device_vector<uint32_t> all_edges;
    std::vector<uint8_t> edge_batch_ids;

    // Second pass: Load and copy scene data
    int vertex_offset = 0;
    // int face_offset = 0;
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
        memcpy(&faces_eptr[face_offset], scene.faces_eptr.data(), (scene_face_count+1)* sizeof(uint32_t));
        std::fill(face_batch_ids.begin() + face_offset, face_batch_ids.begin() + face_offset + scene.nfaces, batch_idx);

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
        // face_offset += scene.nfaces;
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
    printf("Total edges: %d (edge batch ID size: %zu)\n", total_edges, edge_batch_ids.size());
    
    // Note: These appear to be unused in the original code
    thrust::device_vector<uint8_t> unused_gcolors;
    thrust::device_vector<uint32_t> unused_csr_edges;
    thrust::device_vector<uint32_t> unused_csr_eptr;

    PointCloudData data(
        features, positions, faces, faces_eptr, all_edges,
        vertex_batch_ids, edge_batch_ids, face_batch_ids, vertex_ptr, edge_ptr, face_ptr,
        bounding_boxes, batch_size, total_vertices, total_edges, total_faces
    );
    return data;
}



// -- write each scene; [nnodes == spix if point-cloud is the superpixel point cloud] --
bool write_scene(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, PointCloudData& data){
                // float3* ftrs_cu, float3* pos_cu, uint32_t* edges_cu, int* ptr_cu, int* eptr_cu, 
                // uint8_t* gcolor_cu, uint32_t* labels_cu){

    // -- sync before io --
    cudaDeviceSynchronize();

    for(int batch_index=0; batch_index < data.B; batch_index++){

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

    }

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
    faces_eptr.resize(face_count+1);


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
        faces_eptr[face_count] = 3*face_count;
        for (int i = 0; i < face_count; ++i) {
            unsigned char vertex_count_face;
            file.read(reinterpret_cast<char*>(&vertex_count_face), sizeof(unsigned char));
            if (vertex_count_face != 3){
                printf("vertex count: %d\n",vertex_count_face);
            }
            std::vector<int> vertices(vertex_count_face);
            faces_eptr[i] = 3*i;
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
                
                // -- view --
                // bool conda = (a == 295906) && (b == 296744);
                // bool condb = (b == 295906) && (a == 296744);
                // if (conda || condb){
                //     printf("face_index, v0, v1: %d %d %d\n",i, a, b);
                // }

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


bool ScanNetScene::write_ply(const std::filesystem::path& ply_file, PointCloudDataHost& data, bool write_edges, bool write_faces) {

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
    if (!data.edges.empty() && write_edges){
        header += "element edge " + std::to_string(data.E) + '\n';
        header += "property int vertex1\n";
        header += "property int vertex2\n";
    }
    if ((!data.faces.empty()) && write_faces){
        header += "element face " + std::to_string(data.F) + "\n";
        header += "property list uchar int vertex_indices\n";
        if(!data.face_colors.empty()){
            header += "property uchar red\n";
            header += "property uchar green\n";
            header += "property uchar blue\n";
            header += "property uchar alpha\n";
        }
    }
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

    if (!data.edges.empty()){
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
    }  
    
    if ((!data.faces.empty()) && write_faces){
        for (int face = 0; face < data.F; ++face){
            int start = data.faces_eptr[face];
            int end   = data.faces_eptr[face+1];
            unsigned char num = end - start;
            //if (num == 0){ continue; }
            file.write(reinterpret_cast<const char*>(&num), sizeof(unsigned char));

            // bool any_zero = false;
            // for (int index = start; index < end; index++){
            //     any_zero = any_zero || (data.faces[index] == 0);
            // }

            // if (any_zero){
            //     printf("Num: face[%d] %d\n",face,end-start);
            // }
            for (int index = start; index < end; index++){
                int v = data.faces[index];
                // if (any_zero){
                //     printf("face[%d] %d\n",face,v);
                // }
                file.write(reinterpret_cast<const char*>(&v), sizeof(int));
            }

            if(!data.face_colors.empty()){
                uint32_t label = 0;
                if (!data.face_labels.empty()){
                    label = data.face_labels[face];
                }
                unsigned char alpha = (label < UINT32_MAX) ? 255 : 0;
                float3 col = data.face_colors[face];
                unsigned char r = static_cast<unsigned char>(col.x * 255.0f);
                unsigned char g = static_cast<unsigned char>(col.y * 255.0f);
                unsigned char b = static_cast<unsigned char>(col.z * 255.0f);
                file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
                file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
                file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
                file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));


            }
        }
    }  
    

    file.close();
    return true;

};


// -- write each scene; [nnodes == spix if point-cloud is the superpixel point cloud] --
bool write_spix(const std::vector<std::filesystem::path>& scene_files, 
                const std::filesystem::path& output_root, SuperpixelParams3d& spix_params){
   
    // -- sync before io --
    cudaDeviceSynchronize();

    int bx = 0;
    for (const auto& scene_file : scene_files) {

        // -- get batch slice --
        SuperpixelParams3dHost host_spix(spix_params,bx);

        // -- write --
        ScanNetScene scene;
        if(!scene.write_spix_ply_with_fn(scene_file,output_root,host_spix)){
            exit(1);
        }

        bx += 1;
    }

    return 0;
}


bool ScanNetScene::write_spix_ply_with_fn(const std::filesystem::path& scene_path, 
                                  const std::filesystem::path& output_root,
                                  const SuperpixelParams3dHost& data){
                                //   thrust::host_vector<float3>& ftrs, 
                                //   thrust::host_vector<double3>& pos,
                                //   thrust::host_vector<double3>& var, 
                                //   thrust::host_vector<double3>& cov, int nspix) {

    // -- helper --
    std::string line;

    // -- make dir if needed --
    std::string scene_name = scene_path.filename().string();
    std::filesystem::path write_path = output_root / scene_name;
    if (!std::filesystem::exists(write_path)) {
        std::filesystem::create_directories(write_path);
    }
    // -- get filenames --
    std::filesystem::path ply_file = write_path / (scene_name + "_spix.ply");
    std::cout << ply_file << std::endl;
    
    // thrust::host_vector<uint32_t> border_edges(0); // spoof for now.
    // thrust::host_vector<uint32_t> border_ptr(0); // spoof for now.
    // return this->write_spix_ply(ply_file,ftrs,pos,var,cov,nspix);
    return this->write_spix_ply(ply_file,data);

}


bool ScanNetScene::write_spix_ply(const std::filesystem::path& ply_file, const SuperpixelParams3dHost& data){
                                //   thrust::host_vector<float3>& ftrs, 
                                //   thrust::host_vector<double3>& pos,
                                //   thrust::host_vector<double3>& var, 
                                //   thrust::host_vector<double3>& cov, 
                                // //   thrust::host_vector<uint32_t>& border_edges,
                                // //   thrust::host_vector<uint32_t>& border_ptr,
                                //   int nspix) {
    
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
    header += "element vertex " + std::to_string(data.nspix) + "\n";
    header += "property float x\n";
    header += "property float y\n";
    header += "property float z\n";
    header += "property float var_x\n";
    header += "property float var_y\n";
    header += "property float var_z\n";
    header += "property float cov_xy\n";
    header += "property float cov_xz\n";
    header += "property float cov_yz\n";
    header += "property uchar red\n";
    header += "property uchar green\n";
    header += "property uchar blue\n";
    header += "property uchar alpha\n";
    // if (border_edges.size() > 0){
    //     header += "element edge " + std::to_string(border_edges.size()/2) + '\n';
    //     header += "property int vertex1\n";
    //     header += "property int vertex2\n";
    //     // header += "element face " + std::to_string(border_edges.size()/2) + "\n";
    //     // header += "property list uchar int vertex_indices\n";
    // }
    header += "end_header\n";
    file.write(header.c_str(), header.length());

    // -- write data --
    for (int i = 0; i < data.nspix; ++i) {
        
        // -- init --
        unsigned char r, g, b, alpha;
        uint32_t label;
        uint8_t gcolor_id;

        // -- unpack --
        float3 _ftrs = data.mu_app[i];
        double3 _pos = data.mu_pos[i];
        double3 _var = data.var_pos[i];
        double3 _cov = data.cov_pos[i];
        float x = _pos.x;
        float y = _pos.y;
        float z = _pos.z;
        float var_x = _var.x;
        float var_y = _var.y;
        float var_z = _var.z;
        float cov_x = _cov.x;
        float cov_y = _cov.y;
        float cov_z = _cov.z;

        r = static_cast<unsigned char>(_ftrs.x * 255.0f);
        g = static_cast<unsigned char>(_ftrs.y * 255.0f);
        b = static_cast<unsigned char>(_ftrs.z * 255.0f);
        alpha = 255;
        //printf("label: %ld\n",label);
        //printf("x,y,z r,g,b: %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n",x,y,z,ftrs[3*i+0],ftrs[3*i+1] ,ftrs[3*i+2] );


        // -- write --
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&var_x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&var_y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&var_z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&cov_x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&cov_y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&cov_z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
        file.write(reinterpret_cast<const char*>(&alpha), sizeof(unsigned char));
    }


    file.close();
    return true;

};
