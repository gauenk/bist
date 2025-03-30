__host__ int
lots_of_merge(float* img, int* seg,spix_params* sp_params,bool* border,
              spix_helper* sp_helper,spix_helper_sm* sm_helper,
              int* sm_seg1 ,int* sm_seg2, int* sm_pairs,
              int niters, int niters_seg, int sm_start,
              float sigma2_app,  float sigma2_size, int sp_size,
              float potts, float alpha_hastings, float split_alpha, int nspix,
              int nspix_buffer, int nbatch, int width, int height, int nftrs);


std::tuple<int*,bool*,SuperpixelParams*>
  run_lots_of_merge(float* img, int* spix_ptr, int nspix,
                    int nbatch, int height, int width, int nftrs,
                    int niters, int niters_seg, int sm_start, int sp_size,
                    float sigma2_app, float sigma2_size, float potts,
                    float alpha_hastings, float split_alpha);
