#ifndef SHARED_STRUCTS_3D
#define SHARED_STRUCTS_3D

// -- cuda --
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

// -- io --
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>

#include <sstream>
#include <iomanip>

#include <vector>
#include <string>
#include <tuple>
#include <filesystem>



struct alignas(32) spix_helper {
    // -- summary stats --
    double3 sum_app;
    double3 sum_pos;
    double3 sq_sum_self_pos;
    double3 sq_sum_pairs_pos;
};

struct alignas(32) spix_params{
    // -- appearance --
    float3 mu_app;
    // -- shape --
    double3 mu_pos;
    double3 var_pos; // (x,x), (y,y), (z,z)
    double3 cov_pos; // (x,y), (x,z), (y,z)
    double logdet_sigma_shape;
    // -- misc --
    int count;
    bool valid;
};

struct SuperpixelParams3d{

  // -- spix and border --
  thrust::device_vector<uint32_t> spix;
  thrust::device_vector<bool> border;
  thrust::device_vector<bool> is_simple_point;
  thrust::device_vector<uint8_t> neigh_neq;

  // -- summary stats about nspix --
  thrust::device_vector<uint32_t> nspix;
  thrust::device_vector<uint32_t> prev_nspix;
  thrust::device_vector<uint32_t> csum_nspix;

  // -- appearance and shape --
  thrust::device_vector<float3> mu_app;
  thrust::device_vector<double3> mu_pos;
  thrust::device_vector<double3> var_pos;
  thrust::device_vector<double3> cov_pos;

  // -- misc --
  uint32_t nspix_sum = 0;
  int V;
  int B;

  // -- explicit pointer casts --
  uint32_t* spix_ptr() {
      return thrust::raw_pointer_cast(spix.data());
  }
  uint32_t* nspix_ptr() {
      return thrust::raw_pointer_cast(nspix.data());
  }
  uint32_t* csum_nspix_ptr() {
      return thrust::raw_pointer_cast(csum_nspix.data());
  }
  uint32_t* prev_nspix_ptr() {
      return thrust::raw_pointer_cast(prev_nspix.data());
  }
  bool* border_ptr() {
      return thrust::raw_pointer_cast(border.data());
  }
  bool* is_simple_point_ptr() {
      return thrust::raw_pointer_cast(is_simple_point.data());
  }
  uint8_t* neigh_neq_ptr() {
      return thrust::raw_pointer_cast(neigh_neq.data());
  }
  float3* mu_app_ptr(){
    return thrust::raw_pointer_cast(mu_app.data());
  }
  
  
  void comp_csum_nspix(){
    uint32_t* nspix_ptr = thrust::raw_pointer_cast(nspix.data());
    uint32_t* csum_ptr = thrust::raw_pointer_cast(csum_nspix.data());
    thrust::inclusive_scan(thrust::device, nspix_ptr, nspix_ptr + B, csum_ptr + 1);
  }
  uint32_t comp_nspix_sum(){
    uint32_t total = thrust::reduce(nspix.begin(), nspix.end());
    return total;
  }
  void set_nspix_sum(uint32_t in_sum){
    nspix_sum = in_sum;
  }

  void resize_spix_params(uint32_t in_nspix){
    // -- appearance --
    mu_app.resize(in_nspix);
    mu_pos.resize(in_nspix);
    var_pos.resize(in_nspix);
    cov_pos.resize(in_nspix);
  }

  // Constructor to initialize all vectors with the given size
  SuperpixelParams3d(int V, int B) : V(V), B(B) {
    // -- superpixel laeling --
    spix.resize(V);
    border.resize(V);
    is_simple_point.resize(V);
    neigh_neq.resize(V);
    nspix.resize(B,0);
    prev_nspix.resize(B,0);
    csum_nspix.resize(B+1,0);
  }

};

struct SuperpixelParams3dHost {

    // Superpixel stats for this batch
    uint32_t nspix;
    uint32_t prev_nspix;
    uint32_t spix_offset;
    
    // Appearance and shape data for superpixels in this batch
    std::vector<float3> mu_app;
    std::vector<double3> mu_pos;
    std::vector<double3> var_pos;
    std::vector<double3> cov_pos;

    // Alternative constructor if you already have vertex boundaries
    SuperpixelParams3dHost(const SuperpixelParams3d& device_data, int batch_idx) {
        // Validate inputs
        if (batch_idx < 0 || batch_idx >= device_data.B) {
            throw std::invalid_argument("Invalid batch index");
        }

        // Get nspix data to host first
        thrust::host_vector<uint32_t> nspix_host(device_data.nspix.size());
        thrust::copy(device_data.nspix.begin(), device_data.nspix.end(), nspix_host.begin());
        
        thrust::host_vector<uint32_t> prev_nspix_host(device_data.prev_nspix.size());
        thrust::copy(device_data.prev_nspix.begin(), device_data.prev_nspix.end(), prev_nspix_host.begin());
        
        thrust::host_vector<uint32_t> csum_nspix_host(device_data.csum_nspix.size());
        thrust::copy(device_data.csum_nspix.begin(), device_data.csum_nspix.end(), csum_nspix_host.begin());
        
        // Set batch-specific nspix values
        nspix = nspix_host[batch_idx];
        prev_nspix = prev_nspix_host[batch_idx];
        
        // Calculate superpixel offset for this batch
        spix_offset = (batch_idx == 0) ? 0 : csum_nspix_host[batch_idx];
        
        // Extract superpixel-level data for this batch
        if (nspix > 0) {
            int spix_start = spix_offset;
            int spix_end = spix_start + nspix;
            
            // Check if superpixel data arrays are large enough
            if (device_data.mu_app.size() >= spix_end) {
                mu_app.resize(nspix);
                thrust::copy(device_data.mu_app.begin() + spix_start,
                           device_data.mu_app.begin() + spix_end,
                           reinterpret_cast<float3*>(mu_app.data()));
            }
            
            if (device_data.mu_pos.size() >= spix_end) {
                mu_pos.resize(nspix);
                thrust::copy(device_data.mu_pos.begin() + spix_start,
                           device_data.mu_pos.begin() + spix_end,
                           reinterpret_cast<double3*>(mu_pos.data()));
            }
            
            if (device_data.var_pos.size() >= spix_end) {
                var_pos.resize(nspix);
                thrust::copy(device_data.var_pos.begin() + spix_start,
                           device_data.var_pos.begin() + spix_end,
                           reinterpret_cast<double3*>(var_pos.data()));
            }
            
            if (device_data.cov_pos.size() >= spix_end) {
                cov_pos.resize(nspix);
                thrust::copy(device_data.cov_pos.begin() + spix_start,
                           device_data.cov_pos.begin() + spix_end,
                           reinterpret_cast<double3*>(cov_pos.data()));
            }
        }
    }
    
    // Default constructor
    SuperpixelParams3dHost() : nspix(0), prev_nspix(0), spix_offset(0) {}
    
    // // Helper method to get the global superpixel ID range for this batch
    // std::pair<uint32_t, uint32_t> get_spix_range() const {
    //     return {spix_offset, spix_offset + nspix};
    // }
    
    // // Helper method to convert local superpixel ID to global ID
    // uint32_t local_to_global_spix_id(uint32_t local_id) const {
    //     return spix_offset + local_id;
    // }
    
    // // Helper method to convert global superpixel ID to local ID (returns -1 if not in this batch)
    // int global_to_local_spix_id(uint32_t global_id) const {
    //     if (global_id >= spix_offset && global_id < spix_offset + nspix) {
    //         return global_id - spix_offset;
    //     }
    //     return -1;
    // }
};


struct SpixMetaData {
    // Algorithm parameters
    int niters;
    int niters_seg;
    int sm_start;
    int sp_size;
    float sigma2_app;
    float sigma2_size;
    float potts;
    float alpha;
    float split_alpha;
    int target_nspix;
    int nspix_buffer_mult;
    
    // Could be const after initialization
    SpixMetaData(int niters, int niters_seg, int sm_start, int sp_size,
                 float sigma2_app, float sigma2_size, float potts,
                 float alpha, float split_alpha, int target_nspix,
                 int nspix_buffer_mult) :
        niters(niters)
        , niters_seg(niters_seg)
        , sm_start(sm_start)
        , sp_size(sp_size)
        , sigma2_app(sigma2_app)
        , sigma2_size(sigma2_size)
        , potts(potts)
        , alpha(alpha)
        , split_alpha(split_alpha)
        , target_nspix(target_nspix)
        , nspix_buffer_mult(nspix_buffer_mult)
    {}
};



struct PointCloudData {
    // Core data arrays as device_vectors
    thrust::device_vector<float3> ftrs;
    thrust::device_vector<float3> pos;
    thrust::device_vector<uint32_t> faces;
    thrust::device_vector<uint32_t> faces_eptr; // similar to the csr_eptr
    thrust::device_vector<uint32_t> face_labels;
    thrust::device_vector<float3> face_colors;
    thrust::device_vector<uint32_t> face_vnames; // temporary hack for now
    thrust::device_vector<uint32_t> edges;
    thrust::device_vector<uint32_t> csr_edges;
    thrust::device_vector<uint32_t> csr_eptr;
    thrust::device_vector<uint8_t> gcolors;
    thrust::device_vector<uint8_t> vertex_batch_ids;
    thrust::device_vector<uint8_t> edge_batch_ids;
    thrust::device_vector<int> vptr;
    thrust::device_vector<int> eptr;
    thrust::device_vector<int> fptr;
    thrust::device_vector<float> bounding_boxes;
    thrust::device_vector<uint32_t> labels;
    thrust::device_vector<bool> border;
   
    // Scalar parameters
    uint8_t gchrome;
    uint32_t B, V, E, F;
   
    // Helper methods to get raw pointers when needed
    float3* ftrs_ptr() { return thrust::raw_pointer_cast(ftrs.data()); }
    float3* pos_ptr() { return thrust::raw_pointer_cast(pos.data()); }
    uint32_t* faces_ptr() { return thrust::raw_pointer_cast(faces.data()); }
    uint32_t* faces_eptr_ptr() { return thrust::raw_pointer_cast(faces_eptr.data()); }
    uint32_t* edges_ptr() { return thrust::raw_pointer_cast(edges.data()); }
    uint32_t* csr_edges_ptr() { return thrust::raw_pointer_cast(csr_edges.data()); }
    uint32_t* csr_eptr_ptr() { return thrust::raw_pointer_cast(csr_eptr.data()); }
    uint8_t* gcolors_ptr() { return thrust::raw_pointer_cast(gcolors.data()); }
    uint8_t* vertex_batch_ids_ptr() { return thrust::raw_pointer_cast(vertex_batch_ids.data()); }
    uint8_t* edge_batch_ids_ptr() { return thrust::raw_pointer_cast(edge_batch_ids.data()); }
    int* vptr_ptr() { return thrust::raw_pointer_cast(vptr.data()); }
    int* eptr_ptr() { return thrust::raw_pointer_cast(eptr.data()); }
    int* fptr_ptr() { return thrust::raw_pointer_cast(fptr.data()); }
    float* bounding_boxes_ptr() { return thrust::raw_pointer_cast(bounding_boxes.data()); }
    uint32_t* labels_ptr() { return thrust::raw_pointer_cast(labels.data()); }
    bool* border_ptr() { return thrust::raw_pointer_cast(border.data()); }
    uint32_t* face_vnames_ptr() { return thrust::raw_pointer_cast(face_vnames.data()); }
   
    // Host vector methods for data transfer
    thrust::host_vector<float3> ftrs_host() const {
        thrust::host_vector<float3> result(V);
        thrust::copy(ftrs.begin(), ftrs.end(),
                    reinterpret_cast<float3*>(result.data()));
        return result;
    }
    
    thrust::host_vector<float3> pos_host() const {
        thrust::host_vector<float3> result(V);
        thrust::copy(pos.begin(), pos.end(),
                    reinterpret_cast<float3*>(result.data()));
        return result;
    }
    
    thrust::host_vector<uint32_t> faces_host() const {
        thrust::host_vector<uint32_t> result(faces.size());
        thrust::copy(faces.begin(), faces.end(), result.begin());
        return result;
    }

    thrust::host_vector<uint32_t> faces_eptr_host() const {
        thrust::host_vector<uint32_t> result(faces_eptr.size());
        thrust::copy(faces_eptr.begin(), faces_eptr.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<uint32_t> edges_host() const {
        thrust::host_vector<uint32_t> result(edges.size());
        thrust::copy(edges.begin(), edges.end(), result.begin());
        return result;
    }  

    thrust::host_vector<uint32_t> csr_edges_host() const {
        thrust::host_vector<uint32_t> result(csr_edges.size());
        thrust::copy(csr_edges.begin(), csr_edges.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<uint32_t> csr_eptr_host() const {
        thrust::host_vector<uint32_t> result(csr_eptr.size());
        thrust::copy(csr_eptr.begin(), csr_eptr.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<uint8_t> gcolors_host() const {
        thrust::host_vector<uint8_t> result(gcolors.size());
        thrust::copy(gcolors.begin(), gcolors.end(), result.begin());
        return result;
    }

    thrust::host_vector<uint8_t> vertex_batch_ids_host() const {
        thrust::host_vector<uint8_t> result(vertex_batch_ids.size());
        thrust::copy(vertex_batch_ids.begin(), vertex_batch_ids.end(), result.begin());
        return result;
    }

    thrust::host_vector<uint8_t> edge_batch_ids_host() const {
        thrust::host_vector<uint8_t> result(edge_batch_ids.size());
        thrust::copy(edge_batch_ids.begin(), edge_batch_ids.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<int> vptr_host() const {
        thrust::host_vector<int> result(vptr.size());
        thrust::copy(vptr.begin(), vptr.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<int> eptr_host() const {
        thrust::host_vector<int> result(eptr.size());
        thrust::copy(eptr.begin(), eptr.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<int> fptr_host() const {
        thrust::host_vector<int> result(fptr.size());
        thrust::copy(fptr.begin(), fptr.end(), result.begin());
        return result;
    }
    
    thrust::host_vector<float> bounding_boxes_host() const {
        thrust::host_vector<float> result(bounding_boxes.size());
        thrust::copy(bounding_boxes.begin(), bounding_boxes.end(), result.begin());
        return result;
    }

  // Copy constructor
  PointCloudData(const PointCloudData& other)
      : ftrs(other.ftrs)
      , pos(other.pos)
      , faces(other.faces)
      , faces_eptr(other.faces_eptr)
      , face_vnames(other.face_vnames)
      , face_labels(other.face_labels)
      , face_colors(other.face_colors)
      , edges(other.edges)
      , csr_edges(other.csr_edges)
      , csr_eptr(other.csr_eptr)
      , gcolors(other.gcolors)
      , vertex_batch_ids(other.vertex_batch_ids)
      , edge_batch_ids(other.edge_batch_ids)
      , vptr(other.vptr)
      , eptr(other.eptr)
      , fptr(other.fptr)
      , bounding_boxes(other.bounding_boxes)
      , labels(other.labels)
      , gchrome(other.gchrome)
      , B(other.B), V(other.V), E(other.E), F(other.F)
  {}

  PointCloudData copy() const {
    return PointCloudData(*this);
  }

  PointCloudData(const std::vector<float3>& ftrs_h,
                const std::vector<float3>& pos_h,
                const std::vector<uint32_t>& faces_h,
                const std::vector<uint32_t>& faces_eptr_h,
                const thrust::device_vector<uint32_t>& edges_h,
                const std::vector<uint8_t>& vertex_batch_ids_h,
                const std::vector<uint8_t>& edge_batch_ids_h,
                const std::vector<int>& vptr_h,
                const std::vector<int>& eptr_h,
                const std::vector<int>& fptr_h,
                const std::vector<float>& bounding_boxes_h,
                int B, int V, int E, int F)
      : ftrs(ftrs_h.begin(), ftrs_h.end())
      , pos(pos_h.begin(), pos_h.end())
      , faces(faces_h.begin(), faces_h.end())
      , faces_eptr(faces_eptr_h.begin(),faces_eptr_h.end())
      , edges(edges_h)
      , vertex_batch_ids(vertex_batch_ids_h.begin(), vertex_batch_ids_h.end())
      , edge_batch_ids(edge_batch_ids_h.begin(), edge_batch_ids_h.end())
      , vptr(vptr_h.begin(), vptr_h.end())
      , eptr(eptr_h.begin(), eptr_h.end())
      , fptr(fptr_h.begin(), fptr_h.end())
      , bounding_boxes(bounding_boxes_h.begin(), bounding_boxes_h.end())
      , gchrome(0)
      , B(B), V(V), E(E), F(F)
  {}

  PointCloudData(const thrust::device_vector<float3>& ftrs_d,
                const thrust::device_vector<float3>& pos_d,
                const thrust::device_vector<uint32_t>& faces_d,
                const thrust::device_vector<uint32_t>& faces_eptr_d,
                const thrust::device_vector<uint32_t>& edges_d,
                const thrust::device_vector<uint8_t>& vertex_batch_ids_d,
                const thrust::device_vector<uint8_t>& edge_batch_ids_d,
                const thrust::device_vector<int>& vptr_d,
                const thrust::device_vector<int>& eptr_d,
                const thrust::device_vector<int>& fptr_d,
                const thrust::device_vector<float>& bounding_boxes_d,
                int B, int V, int E, int F)
      : ftrs(ftrs_d)
      , pos(pos_d)
      , faces(faces_d)
      , faces_eptr(faces_eptr_d)
      , edges(edges_d)
      , vertex_batch_ids(vertex_batch_ids_d)
      , edge_batch_ids(edge_batch_ids_d)
      , vptr(vptr_d)
      , eptr(eptr_d)
      , fptr(fptr_d)
      , bounding_boxes(bounding_boxes_d)
      , gchrome(0)
      , B(B), V(V), E(E), F(F)
  {}

};


// Warning; batching problems. the idea of "csr_edges" uses V+1 elements but the vptr just gives "num of vertices in batch".
// but maybe okay if "_start" is the "+1" of the previous? Or we know to write "+1"? weird..
// 
// After a bit of thought, there is not problem. vptr[bx] is the STARTING index of batch "bx"... so effectively it is already "V+1"
//

struct PointCloudDataHost {
    // Host data arrays
    std::vector<float3> ftrs;
    std::vector<float3> pos;
    std::vector<uint32_t> faces;
    std::vector<uint32_t> faces_eptr;
    std::vector<uint32_t> face_labels;
    std::vector<float3> face_colors;
    std::vector<uint32_t> edges;
    std::vector<uint32_t> csr_edges;
    std::vector<uint32_t> csr_eptr;
    std::vector<uint8_t> gcolors;
    std::vector<uint8_t> vertex_batch_ids;
    std::vector<uint8_t> edge_batch_ids;
    std::vector<float> bounding_boxes;
    std::vector<uint32_t> labels;
    
    // Scalar parameters for this batch
    uint8_t gchrome;
    int V, E, F;  // sizes for this specific batch
    
    // Constructor that extracts a single batch from device data
    PointCloudDataHost(const PointCloudData& device_data, int batch_idx) {
        // Validate batch index
        if (batch_idx < 0 || batch_idx >= device_data.B) {
            throw std::invalid_argument("Invalid batch index");
        }
        
        // Copy scalar data
        gchrome = device_data.gchrome;
        
        // Get batch boundaries
        thrust::host_vector<int> vptr_host = device_data.vptr_host();
        thrust::host_vector<int> eptr_host = device_data.eptr_host();
        thrust::host_vector<int> fptr_host = device_data.fptr_host();
        
        int v_start = vptr_host[batch_idx];
        int v_end = vptr_host[batch_idx + 1];
        int e_start = eptr_host[batch_idx];
        int e_end = eptr_host[batch_idx + 1];
        int f_start = fptr_host[batch_idx];
        int f_end = fptr_host[batch_idx + 1];
        
        // Set batch sizes
        V = v_end - v_start;
        E = e_end - e_start;
        F = f_end - f_start;
        
        // Extract vertex data
        if (V > 0) {
            ftrs.resize(V);
            pos.resize(V);
            
            thrust::copy(device_data.ftrs.begin() + v_start,
                        device_data.ftrs.begin() + v_end,
                        reinterpret_cast<float3*>(ftrs.data()));
                        
            thrust::copy(device_data.pos.begin() + v_start,
                        device_data.pos.begin() + v_end,
                        reinterpret_cast<float3*>(pos.data()));
            
            // Extract vertex batch IDs if they exist
            if (!device_data.vertex_batch_ids.empty()) {
                vertex_batch_ids.resize(V);
                thrust::copy(device_data.vertex_batch_ids.begin() + v_start,
                           device_data.vertex_batch_ids.begin() + v_end,
                           vertex_batch_ids.begin());
            }
            
            // Extract colors if they exist
            if (!device_data.gcolors.empty()) {
                gcolors.resize(V);
                thrust::copy(device_data.gcolors.begin() + v_start,
                           device_data.gcolors.begin() + v_end,
                           gcolors.begin());
            }
            
            // Extract labels if they exist
            if (!device_data.labels.empty()) {
                labels.resize(V);
                thrust::copy(device_data.labels.begin() + v_start,
                           device_data.labels.begin() + v_end,
                           labels.begin());
            }
        }
        
        // Extract edge data
        if (E > 0) {
            edges.resize(2*E);
            thrust::copy(device_data.edges.begin() + 2*e_start,
                        device_data.edges.begin() + 2*e_end,
                        edges.begin());
            
            // Extract edge batch IDs if they exist
            if (!device_data.edge_batch_ids.empty()) {
                edge_batch_ids.resize(E);
                thrust::copy(device_data.edge_batch_ids.begin() + e_start,
                           device_data.edge_batch_ids.begin() + e_end,
                           edge_batch_ids.begin());
            }
        }
        
        // Extract face data [warning; batching problems with extraction; "F+1" vs "F" for each batch... maybe not a problem...]
        if ((F > 0) && (device_data.faces_eptr.size()>0)) {
            faces_eptr.resize(F+1);
            printf("%d %d\n",F,device_data.faces_eptr.size());
            thrust::copy(device_data.faces_eptr.begin() + f_start,
                        device_data.faces_eptr.begin() + F+1,
                        faces_eptr.begin());
            uint32_t start = faces_eptr[0];
            uint32_t end = faces_eptr[F];
            uint32_t size = end - start;
            //printf("start,end: %d %d\n",start,end);
            faces.resize(size);
            thrust::copy(device_data.faces.begin() + start,
                        device_data.faces.begin() + end,
                        faces.begin());
            if(!device_data.face_labels.empty()){
                face_labels.resize(F);
                thrust::copy(device_data.face_labels.begin() + f_start,
                             device_data.face_labels.begin() + F,
                             face_labels.begin());
            }
            if(!device_data.face_colors.empty()){
                face_colors.resize(F);
                thrust::copy(device_data.face_colors.begin() + f_start,
                             device_data.face_colors.begin() + F,
                             face_colors.begin());
            }

        }
        
        // Extract CSR edges if they exist
        if (!device_data.csr_edges.empty()) {
            // For CSR format, we need to be more careful about extracting the right portion
            // This is a simplified extraction - you might need to adjust based on your CSR structure
            csr_edges.resize(E);
            thrust::copy(device_data.csr_edges.begin() + e_start,
                        device_data.csr_edges.begin() + e_end,
                        csr_edges.begin());
        }
        
        // Extract CSR edge pointers if they exist
        if (!device_data.csr_eptr.empty()) {
            csr_eptr.resize(V + 1);
            thrust::copy(device_data.csr_eptr.begin() + v_start,
                        device_data.csr_eptr.begin() + v_end + 1,
                        csr_eptr.begin());
        }
        
        // Extract bounding boxes if they exist (assuming 6 floats per batch: min_x, min_y, min_z, max_x, max_y, max_z)
        if (!device_data.bounding_boxes.empty()) {
            int bb_start = batch_idx * 6;
            int bb_end = bb_start + 6;
            bounding_boxes.resize(6);
            thrust::copy(device_data.bounding_boxes.begin() + bb_start,
                        device_data.bounding_boxes.begin() + bb_end,
                        bounding_boxes.begin());
        }
    }
    
    // Default constructor
    PointCloudDataHost() : gchrome(0), V(0), E(0), F(0) {}
};


  //   PointCloudData(const thrust::host_vector<float3>& ftrs_h,
  //               const thrust::host_vector<float3>& pos_h,
  //               const thrust::host_vector<uint32_t>& faces_h,
  //               const thrust::host_vector<uint32_t>& edges_h,
  //               // const thrust::host_vector<uint32_t>& csr_edges_h,
  //               // const thrust::host_vector<uint32_t>& csr_eptr_h,
  //               // const thrust::host_vector<uint8_t>& gcolors_h,
  //               const thrust::host_vector<uint8_t>& bids_h,
  //               const thrust::host_vector<int>& ptr_h,
  //               const thrust::host_vector<int>& eptr_h,
  //               const thrust::host_vector<int>& fptr_h,
  //               const thrust::host_vector<float>& dim_sizes_h,
  //               uint8_t gchrome, int B, int V, int E, int F)
  //     : ftrs(ftrs_h.begin(), ftrs_h.end())
  //     , pos(pos_h.begin(), pos_h.end())
  //     , faces(faces_h.begin(), faces_h.end())
  //     , edges(edges_h.begin(), edges_h.end())
  //     // , csr_edges(csr_edges_h.begin(), csr_edges_h.end())
  //     // , csr_eptr(csr_eptr_h.begin(), csr_eptr_h.end())
  //     // , gcolors(gcolors_h.begin(), gcolors_h.end())
  //     , bids(bids_h.begin(), bids_h.end())
  //     , ptr(ptr_h.begin(), ptr_h.end())
  //     , eptr(eptr_h.begin(), eptr_h.end())
  //     , fptr(fptr_h.begin(), fptr_h.end())
  //     , dim_sizes(dim_sizes_h.begin(), dim_sizes_h.end())
  //     , gchrome(gchrome)
  //     , B(B), V(V), E(E), F(F)
  // {}

// };

// struct PointCloudData {
//     // Core data arrays as device_vectors
//     thrust::device_vector<float3> ftrs;
//     thrust::device_vector<float3> pos;
//     thrust::device_vector<uint32_t> faces;
//     thrust::device_vector<uint8_t> gcolors;
//     thrust::device_vector<uint32_t> csr_edges;
//     thrust::device_vector<uint32_t> csr_eptr;
//     thrust::device_vector<uint8_t> bids;
//     thrust::device_vector<int> ptr;
//     thrust::device_vector<int> eptr;
//     thrust::device_vector<int> fptr;
//     thrust::device_vector<float> dim_sizes;
    
//     // Scalar parameters
//     uint8_t gchrome;
//     int B, V, E, F;
    
//     // Helper methods to get raw pointers when needed
//     float3* ftrs_ptr() { return thrust::raw_pointer_cast(ftrs.data()); }
//     float3* pos_ptr() { return thrust::raw_pointer_cast(pos.data()); }
//     uint8_t* gcolors_ptr() { return thrust::raw_pointer_cast(gcolors.data()); }
//     // ... other pointer getters
    
//     // Host vector methods become much simpler
//     thrust::host_vector<float> ftrs_host() {
//         thrust::host_vector<float> result(3 * V);
//         thrust::copy(ftrs.begin(), ftrs.end(), 
//                     reinterpret_cast<float*>(result.data()));
//         return result;
//     }
    
//     // Constructor
//     PointCloudData(thrust::device_vector<float3>&& ftrs,
//                    thrust::device_vector<float3>&& pos,
//                    thrust::device_vector<uint8_t>&& gcolors,
//                    // ... other vectors
//                    uint8_t gchrome, int B, int V, int E, int F)
//         : ftrs(std::move(ftrs))
//         , pos(std::move(pos))
//         , gcolors(std::move(gcolors))
//         // ... move other vectors
//         , gchrome(gchrome)
//         , B(B), V(V), E(E), F(F)
//     {
//     }
// };


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



class Logger {

public:
    int bndy_ix = 0;
    std::vector<std::filesystem::path> log_roots;
  Logger(const std::filesystem::path& _output_root,
        std::vector<std::filesystem::path> scene_paths);
  void boundary_update(PointCloudData& data, SuperpixelParams3d& params, spix_params* sp_params);
};



#endif
