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

/*********************************

  -=-=-=-    Float3   -=-=-=-=-

**********************************/


// struct alignas(16) spix_params{
//     // -- appearance --
//     float3 mu_app;
//     /* float3 sigma_app; */
//     float3 prior_mu_app;
//     /* float3 prior_sigma_app; */
//     int prior_mu_app_count; // kappa (kappa_app)
//     /* int prior_sigma_app_count; // nu */
//     // -- shape --
//     double2 mu_shape;
//     double3 sigma_shape;
//     double2 prior_mu_shape;
//     double3 prior_sigma_shape;
//     double3 sample_sigma_shape;
//     int prior_mu_shape_count; // lambda term; (prior mu count)
//     int prior_sigma_shape_count; // nu term; (prior shape count)
//     double3 prior_icov;
//     // double3 prior_icov_eig;
//     // -- helpers --
//     /* double logdet_sigma_app; */
//     double logdet_sigma_shape;
//     double logdet_prior_sigma_shape;
//     // -- priors --
//     double prior_lprob;
//     /* double prior_mu_i_lprob; */
//     /* double prior_sigma_i_lprob; */
//     /* double prior_mu_s_lprob; */
//     /* double prior_sigma_s_lprob; */
//     // -- helpers --
//     int count;
//     float icount; // invalid count; to set the prior for split
//     int sm_count;
//     float prior_count; // df and lam for shape and appearance
//     int valid;
//     bool prop;
// };

// struct alignas(16) spix_helper{
//     float3 sum_app;
//     double3 sq_sum_app;
//     longlong2 sum_shape;
//     longlong3 sq_sum_shape;
// };

// struct alignas(16) spix_helper_sm {
//     double3 sum_app;
//     double3 sq_sum_app;
//     longlong2 sum_shape;
//     // int2 sum_shape;
//     longlong3 sq_sum_shape;
//     /* float3 squares_i; */
//     int count_f;
//     /* float3 b_n; */
//     /* float3 b_n_f; */
//     float3 b_n_app;
//     float3 b_n_f_app;
//     /* float3 b_n_shape; */
//     /* float3 b_n_shape_f; */
//     float b_n_shape_det;
//     float b_n_f_shape_det;
//     /* float3 numerator; */
//     float numerator_app;
//     float numerator_f_app;
//     float3 denominator;
//     float3 denominator_f;
//     // -- ... --
//     float lprob_shape;
//     // -- ... --
//     float lprob_f;
//     float lprob_s_cond;
//     float lprob_s_ucond;
//     float lprob_k_cond;
//     float lprob_k_ucond;
//     // -- [app; for dev] --
//     float lprob_f_app;
//     float lprob_s_cond_app;
//     float lprob_k_cond_app;
//     // -- ... --
//     float hasting;
//     bool select;
//     bool merge; // a bool
//     bool remove;
//     bool stop_bfs;
//     int count;
//     int max_sp;
// };


// struct alignas(16) spix_helper_sm_v2 {
//     // -- summary stats --
//     double3 sum_app;
//     double3 sq_sum_app;
//     longlong2 sum_shape;
//     longlong3 sq_sum_shape;
//     // -- computed stats --
//     double3 sigma_s;
//     double3 sigma_k;
//     double3 sigma_f;
//     // -- whatever --
//     float3 b_n_app;
//     float3 b_n_f_app;
//     float b_n_shape_det;
//     float b_n_f_shape_det;
//     float numerator_app;
//     float numerator_f_app;
//     float3 denominator;
//     float3 denominator_f;
//     // -- app --
//     float lprob_k_cond_app;
//     float lprob_k_ucond_app;
//     float lprob_s_cond_app;
//     float lprob_s_ucond_app;
//     float lprob_f_app;
//     // -- shape --
//     float lprob_k_cond_shape;
//     float lprob_k_ucond_shape;
//     float lprob_s_cond_shape;
//     float lprob_s_ucond_shape;
//     float lprob_f_shape;
//     // -- helpers --
//     float hasting;
//     bool merge; // a bool
//     bool select;
//     bool remove;
//     int count;
//     int ninvalid;
//     int max_sp;
// };


// struct Helpers3d{
//     sp_params
//     SuperpixelParams3d(size_t size) {

//     }
// }

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
    int major_ix = 0;
    int minor_ix = 0;
    int npix = 0;
    int height = 0;
    int width = 0;
    int nspix = 0;
    int niters = 0;
    int niters_seg = 0;
    int bndy_ix = 0;
    int split_ix = 0;
    int merge_ix = 0;
    int relabel_ix = 0;
    int filled_ix = 0;
    int shift_ix = 0;
    int frame_index = 0;
    int merge_details_ix = 0;
    thrust::device_vector<int> spix;
    thrust::device_vector<int> split_prop;
    thrust::device_vector<int> merge_prop;
    thrust::device_vector<float> split_hastings;
    thrust::device_vector<float> merge_hastings;
    thrust::device_vector<int> relabel;
    thrust::device_vector<int> filled;
    const std::string save_directory;

  Logger(const std::string& directory, int frame_index, int height, int width, int nspix,int niters,int niters_seg);
//   void save();
//   template <typename T>
//   void save_seq(const std::string& directory, const thrust::device_vector<T>& device_vec);
//   template <typename T>
//   void save_seq_v2(const std::string& directory, const thrust::device_vector<T>& device_vec, int nftrs);
//   template <typename T>
//   void save_to_csv(const thrust::device_vector<T>& device_vec, const std::string& filename, int h, int w);
//   void save_shifted_spix(int* spix);
//   void boundary_update(int* seg);
//   void log_filled(int* seg);
//   void log_split(int* sm_seg1, int* sm_seg2);
//   void log_merge(int* seg, int* sm_pairs, int nspix);
//   void log_merge_details(int* sm_pairs,spix_params* sp_params,
//                          spix_helper_sm_v2* sm_helper,
//                          const int nspix_buffer, float alpha, float merge_alpha);
//   // void log_relabel(uint64_t* comparisons, float* ss_comps, bool* is_living,
//   //                  int* relabel_id, float epsilon_new, float epsilon_reid, int nspix);
//   void log_relabel(int* spix);
//   template <typename T>
//   void save_seq_frame(const std::string& directory, const thrust::device_vector<T>& img,
//                       int frame_index, int height, int width);
//   std::string get_filename(const std::string& directory, int seq_index);

};






#endif
