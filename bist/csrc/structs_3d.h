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
    // Core data arrays
    float3* ftrs;           // Features
    float3* pos;            // Positions
    uint8_t* gcolors;       // Global colors
    uint32_t* csr_edges;    // CSR format edges
    uint8_t* bids;          // Batch IDs
    int* ptr;               // Pointer array
    uint32_t* csr_eptr;     // CSR edge pointers
    float* dim_sizes;       // Dimension sizes
    
    // Scalar parameters
    uint8_t gchrome;        // Global chrome value
    int B;
    int V;
    int E;
    
    // Constructor
    PointCloudData(float3* ftrs, float3* pos, uint8_t* gcolors,
                   uint32_t* csr_edges, uint8_t* bids, int* ptr, 
                   uint32_t* csr_eptr, float* dim_sizes, 
                   uint8_t gchrome, int B, int V, int E)
        : ftrs(ftrs)
        , pos(pos)
        , gcolors(gcolors)
        , csr_edges(csr_edges)
        , bids(bids)
        , ptr(ptr)
        , csr_eptr(csr_eptr)
        , dim_sizes(dim_sizes)
        , gchrome(gchrome)
        , B(B)
        , V(V)
        , E(E)
    {
    }
};

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
