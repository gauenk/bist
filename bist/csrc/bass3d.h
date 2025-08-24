std::tuple<long*,bool*,SuperpixelParams*>
run_bass3d(float* ftrs, int* pos, int* ptr, int* nnodes, 
           int* dim_sizes, int nbatch, int nftrs,
           int niters, int niters_seg, int sm_start, int sp_size,
           float sigma2_app, float sigma2_size, float potts,
           float alpha_hastings, float split_alpha, int target_nspix, Logger* logger);