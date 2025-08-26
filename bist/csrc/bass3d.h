std::tuple<uint32_t*,bool*,SuperpixelParams*>
run_bass3d(float3* ftrs, float3* pos, uint32_t* edges, uint8_t* bids, 
           int* ptr, int* eptr, float* dim_sizes, int nbatch,
           int niters, int niters_seg, int sm_start, int sp_size,
           float sigma2_app, float sigma2_size, float potts,
           float alpha_hastings, float split_alpha, int target_nspix, Logger* logger=nullptr);