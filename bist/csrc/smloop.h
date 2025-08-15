
int* run_smloop(float* img, int* init_spix, int init_nspix,
                int nbatch, int height, int width, int nftrs,
                int niters, int sp_size, float sigma2_app,
                float alpha_hastings, float potts,
                float merge_alpha, float split_alpha, Logger* logger=nullptr);
