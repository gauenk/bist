
#include "structs.h"

#define THREADS_PER_BLOCK 512

// Constructor to initialize all vectors with the given size
Logger::Logger(const std::string& directory, int frame_index,
               int height, int width, int nspix, int niters, int niters_seg)
    : save_directory(directory), frame_index(frame_index), height(height), width(width),
      nspix(nspix), npix(height * width),
      niters(niters), niters_seg(niters_seg),
      spix(npix),
      // spix(npix * niters * niters_seg),
      // filled(npix * 30),
      // split_prop(2 * npix * niters / 4),
      filled(npix),
      split_prop(npix),
      merge_prop(npix),
      split_hastings(1),
      merge_hastings(1),
      relabel(npix)
{
  if (access(directory.c_str(), F_OK) == -1) { // Check if directory exists
    if (mkdir(directory.c_str(), 0755) != 0) {
      throw std::runtime_error("Failed to create directory: " + directory);
    }
  }
}


// Copy two int* arrays (sm_seg1, sm_seg2) into split_prop
void Logger::boundary_update(int* seg) {

    if (spix.size() < npix) {
      throw std::runtime_error("spix is too small");
    }
    // int start_ix = (major_ix*niters_seg+minor_ix) * npix;
    // // printf("major_ix,minor_ix,major_ix*niters_seg+minor_ix: %d,%d,%d\n",
    // //        major_ix,minor_ix,major_ix*niters_seg+minor_ix);
    // thrust::copy(seg, seg + npix, spix.begin() + start_ix);
    // minor_ix = (minor_ix + 1)%niters_seg;

    // -- copy --
    thrust::copy(seg, seg + npix, spix.begin());

    // -- save --
    std::ostringstream save_dir;
    save_dir << save_directory << "bndy/";
    save_seq_frame(save_dir.str(),spix,bndy_ix,height,width);

    // -- increment counter --
    bndy_ix++;

}

void Logger::log_filled(int* seg) {
    if (spix.size() < npix) {
      throw std::runtime_error("spix is too small");
    }
    // if (filled_ix>=30){ return; }
    // int start_ix = filled_ix * npix;
    // thrust::copy(seg, seg + npix, filled.begin() + start_ix);

    // -- copy & save --
    thrust::copy(seg, seg + npix, filled.begin());
    std::ostringstream save_dir;
    save_dir << save_directory << "filled/";
    save_seq_frame(save_dir.str(),filled,filled_ix,height,width);
    filled_ix++;
}

// Copy two int* arrays (sm_seg1, sm_seg2) into split_prop
void Logger::log_split(int* sm_seg1, int* sm_seg2) {
    // if (split_prop.size() < 2 * npix) {
    //     throw std::runtime_error("split_prop is too small");
    // }
    // int start_ix = 2 * (major_ix/4) * npix;
    // printf("major_ix, start_ix: %d,%d\n",major_ix,start_ix);
    // thrust::copy(sm_seg1, sm_seg1 + npix, split_prop.begin() + start_ix);
    // thrust::copy(sm_seg2, sm_seg2 + npix, split_prop.begin() + start_ix + npix);

    // -- copy & save --
    // std::ostringstream split_dir;
    // split_dir << save_directory << "split/";
    // save_seq(split_dir.str(),split_prop);

    // -- copy & save --
    thrust::copy(sm_seg1, sm_seg1 + npix, split_prop.begin());
    std::ostringstream save_dir;
    save_dir << save_directory << "split/";
    save_seq_frame(save_dir.str(),split_prop,split_ix,height,width);
    split_ix++;

    // -- copy & save --
    thrust::copy(sm_seg2, sm_seg2 + npix, split_prop.begin());
    save_seq_frame(save_dir.str(),split_prop,split_ix,height,width);
    split_ix++;
}



__global__
void log_merge_kernel(int* seg, int* sm_pairs, int* logging_spix, int npix, int width){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;  
  if (idx>=npix) return; 
  int x = idx % width;
  int y = idx / width;
  int C = seg[idx]; // center 
  int prop = sm_pairs[2*C+1];
  if (prop >= 0){
    logging_spix[idx] = prop;
  }else{
    logging_spix[idx] = C;
  }
}

__global__
void log_merge_pairs_kernel(int* sm_pairs, int* merge_pairs, int nspix){
  int spix_id = threadIdx.x + blockIdx.x * blockDim.x;  
  if (spix_id>=nspix) return; 
  merge_pairs[spix_id] = sm_pairs[2*spix_id+1];
}

__global__ void log_merge_details_kernel(int* sm_pairs,
                                         spix_params* sp_params,
                                         spix_helper_sm_v2* sm_helper,
                                         const int nspix_buffer,
                                         float alpha, float merge_alpha, float* log) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    int f = sm_pairs[2*k+1];
    // printf("%d,%d\n",k,f);

    if(f<0) return;
    // printf("%d,%d\n",f,sp_params[f].valid == 0 ? 1: 0);
	if (sp_params[f].valid == 0) return;
    // if(f<=0) return;


    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sm_helper[k].count);
    
    if ((count_k<1)||(count_f<1)) return;

    // sm_helper[k].merge = false;
    float num_k = __ldg(&sm_helper[k].numerator_app);

    float total_marginal_1 = (num_k - __ldg(&sm_helper[k].denominator.x)) +  
                         (num_k - __ldg(&sm_helper[k].denominator.y)) + 
                         (num_k - __ldg(&sm_helper[k].denominator.z)); 

    float num_f = __ldg(&sm_helper[f].numerator_app);

    float total_marginal_2 = (num_f - __ldg(&sm_helper[f].denominator.x)) +   
                         (num_f - __ldg(&sm_helper[f].denominator.y)) + 
                         (num_f - __ldg(&sm_helper[f].denominator.z));

    float num_kf = __ldg(&sm_helper[k].numerator_f_app);

    float total_marginal_f = (num_kf - __ldg(&sm_helper[k].denominator_f.x)) +   
                         (num_kf - __ldg(&sm_helper[k].denominator_f.y)) + 
                         (num_kf - __ldg(&sm_helper[k].denominator_f.z));


    float log_nominator = lgammaf(count_f) + total_marginal_f + lgammaf(alpha) + 
        lgammaf(alpha / 2 + count_k) + lgammaf(alpha / 2 + count_f -  count_k);

   float log_denominator = __logf(alpha) + lgammaf(count_k) + lgammaf(count_f -  count_k) + total_marginal_1 + 
        total_marginal_2 + lgammaf(alpha + count_f) + lgammaf(alpha / 2) + 
        lgammaf(alpha / 2);

    log_denominator = __logf(alpha) + total_marginal_1 + total_marginal_2;
    log_nominator = total_marginal_f ;
    // sm_helper[k].hasting = log_nominator - log_denominator + merge_alpha;

    log[6*k] = log_denominator;
    log[6*k+1] = log_nominator;
    log[6*k+2] = log_nominator - log_denominator + merge_alpha;
    log[6*k+3] = total_marginal_f;
    log[6*k+4] = total_marginal_1;
    log[6*k+5] = total_marginal_2;

}



// -- merge --
void Logger::log_merge_details(int* sm_pairs,spix_params* sp_params,
                               spix_helper_sm_v2* sm_helper,
                               const int nspix_buffer, float alpha, float merge_alpha){

    thrust::device_vector<float> log(nspix_buffer*6);
    float* log_ptr = thrust::raw_pointer_cast(log.data());
    int num_block = ceil( double(nspix_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid(num_block,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    log_merge_details_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sm_pairs,sp_params,sm_helper,
                                                               nspix_buffer,alpha,merge_alpha,log_ptr);

    // -- copy and save --
    std::ostringstream save_dir;
    save_dir << save_directory << "merge_details/";
    save_seq_frame(save_dir.str(),log,merge_details_ix,nspix_buffer,6);
    merge_details_ix++;

}


// -- merge --
void Logger::log_merge(int* seg, int* sm_pairs, int nspix) {
    if (merge_prop.size() < npix) {
      throw std::runtime_error("spix is too small");
    }

    // -- init --
    // int start_ix = 2 * (major_ix/4) * npix;
    // int* logging_spix = thrust::raw_pointer_cast(merge_prop.data()) + start_ix;
    // int* logging_spix_1 = thrust::raw_pointer_cast(merge_prop.data()) + start_ix+npix;

    // -- copy --
    // cudaMemcpy(logging_spix,seg,npix*sizeof(int),cudaMemcpyDeviceToDevice);
    // cudaMemcpy(logging_spix_1,seg,npix*sizeof(int),cudaMemcpyDeviceToDevice);

    // printf("merge start.\n");
    // -- copy and save --
    int* logging_spix = thrust::raw_pointer_cast(merge_prop.data());
    cudaMemcpy(logging_spix,seg,npix*sizeof(int),cudaMemcpyDeviceToDevice);
    std::ostringstream save_dir;
    save_dir << save_directory << "merge/";
    save_seq_frame(save_dir.str(),merge_prop,merge_ix,height,width);
    merge_ix++;

    // -- store merge --
    // int num_block = ceil( double(npix) / double(THREADS_PER_BLOCK) ); 
    // dim3 BlockPerGrid(num_block,1);
    // dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    // log_merge_kernel<<<BlockPerGrid,ThreadPerBlock>>>(seg, sm_pairs, logging_spix, npix, width);

    // -- save --
    // std::ostringstream save_dir;
    // save_dir << save_directory << "merge/";
    // save_seq_frame(save_dir.str(),merge_prop,merge_ix,height,width);

    // -- copy proposed merge spix --
    thrust::device_vector<int> merge_pairs(nspix);
    int* merge_pairs_ptr = thrust::raw_pointer_cast(merge_pairs.data());
    int num_block = ceil( double(nspix) / double(THREADS_PER_BLOCK) ); 
    dim3 BlockPerGrid(num_block,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    log_merge_pairs_kernel<<<BlockPerGrid,ThreadPerBlock>>>(sm_pairs, merge_pairs_ptr, nspix);
    save_seq_frame(save_dir.str(),merge_pairs,merge_ix,nspix,1);
    merge_ix++;
    // printf("merge end.\n");

}



__global__
void log_relabel_kernel(uint64_t* comparisons, float* ss_comps,
                        bool* is_living,  int* relabel_id, float* logging,
                        float epsilon_new, float epsilon_reid, int nspix){


  // -- get superpixel index --
  int spix_id = threadIdx.x + blockIdx.x*blockDim.x;
  if (spix_id>=nspix) return;

  // -- only keep valid --
  bool spix_is_alive = is_living[spix_id];
  if (not(spix_is_alive)){ return; }

  // -- decode --
  float ss_delta = ss_comps[spix_id];

  // -- decode --
  uint64_t comparison = comparisons[spix_id];
  uint32_t delta32 = uint32_t(comparison>>32);
  int candidate_spix = *reinterpret_cast<int*>(&comparison);
  float delta = *reinterpret_cast<float*>(&delta32);

  // -- revive? --
  bool cond_revive = (delta < epsilon_reid);
  cond_revive = cond_revive and (candidate_spix < nspix);
  cond_revive = cond_revive and (candidate_spix != spix_id);

  // -- save --
  float* logging_ix = logging+6*spix_id;
  logging_ix[0] = ss_delta;
  logging_ix[1] = 1.0*candidate_spix;
  logging_ix[2] = delta;
  logging_ix[3] = cond_revive ? 1.0 : 0.0;
  logging_ix[4] = (ss_delta > epsilon_new) ? 1.0 : 0.0;
  logging_ix[5] = relabel_id[spix_id];

}

// -- relabel --
void Logger::log_relabel(int* spix){

  // thrust::device_vector<int> relabel(npix);
  int* relabel_ptr = thrust::raw_pointer_cast(relabel.data());
  cudaMemcpy(relabel_ptr,spix,npix*sizeof(int),cudaMemcpyDeviceToDevice);
  std::ostringstream save_dir;
  save_dir << save_directory << "relabel/";
  save_seq_frame(save_dir.str(),relabel,relabel_ix,height,width);
  relabel_ix++;

}


// void Logger::log_relabel_old(uint64_t* comparisons, float* ss_comps, bool* is_living,
//                          int* relabel_id, float epsilon_new, float epsilon_reid, int max_spix){

//     // -- ... --
//     thrust::device_vector<float> relabel(6*max_spix);
//     // if (relabel.size() < (6*max_spix)) {
//     //   throw std::runtime_error("spix is too small");
//     // }
//     // printf("nspix,max_spix: %d,%d\n",nspix,max_spix);

//     // -- init --
//     // int start_ix = 6 * nspix * relabel_ix;
//     // int end_ix = start_ix + 6 * nspix * relabel_ix;
//     // thrust::fill(relabel.begin()+start_ix,relabel.begin()+end_ix,-1);
//     // float* logging_relabel = thrust::raw_pointer_cast(relabel.data()) + start_ix;
//     float* logging_relabel = thrust::raw_pointer_cast(relabel.data());


//     // -- store merge --
//     int num_block = ceil( double(max_spix) / double(THREADS_PER_BLOCK) ); 
//     dim3 BlockPerGrid(num_block,1);
//     dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
//     log_relabel_kernel<<<BlockPerGrid,ThreadPerBlock>>>(comparisons, ss_comps, is_living,
//                                                         relabel_id, logging_relabel,
//                                                         epsilon_new, epsilon_reid, max_spix);

//     // -- save --
//     std::ostringstream save_dir;
//     save_dir << save_directory << "relabel/";
//     save_seq_frame(save_dir.str(),relabel,relabel_ix,nspix,6);

//     // -- increment counter --
//     relabel_ix++;
// }




// void Logger::copy_merge(int* sm_seg1, int* sm_seg2) {
//     if (merge_prop.size() < 2 * npix) {
//         throw std::runtime_error("merge_prop is too small");
//     }
//     int start_ix = 2 * (major_ix/4) * npix;
//     // printf("major_ix, start_ix: %d,%d\n",major_ix,start_ix);
//     thrust::copy(sm_seg1, sm_seg1 + npix, merge_prop.begin() + start_ix);
//     thrust::copy(sm_seg2, sm_seg2 + npix, merge_prop.begin() + start_ix + npix);
// }


// Save all frames in split_prop
void Logger::save() {

  // -- bndy_updates --
  // std::ostringstream bndy_dir;
  // bndy_dir << save_directory << "bndy/";
  // save_seq(bndy_dir.str(),spix);

  // -- merge --
  // std::ostringstream merge_dir;
  // merge_dir << save_directory << "merge/";
  // save_seq(merge_dir.str(),merge_prop);

  // -- split --
  // std::ostringstream split_dir;
  // split_dir << save_directory << "split/";
  // save_seq(split_dir.str(),split_prop);

  // // -- relabeling --
  // std::ostringstream relabel_dir;
  // relabel_dir << save_directory << "relabel/";
  // save_seq_v2(relabel_dir.str(),relabel,6);

}

// Save all frames in split_prop
void Logger::save_shifted_spix(int* spix_raw) {
  
  // -- offset by index --
  thrust::device_ptr<int> spix_ptr = thrust::device_pointer_cast(spix_raw);
  thrust::device_vector<float> spix(spix_ptr, spix_ptr+npix);

  // -- create directory --
  std::ostringstream directory;
  directory << save_directory << "shifted/";
  auto filename = get_filename(directory.str(),shift_ix);

  // std::string dir_path = directory.str(); // Store as a string
  // if (access(dir_path.c_str(), F_OK) == -1) { // Check if directory exists
  //   if (mkdir(dir_path.c_str(), 0755) != 0) {
  //     throw std::runtime_error("Failed to create directory: " + dir_path);
  //   }
  // }
  // // -- save --
  // std::ostringstream filename;
  // filename << dir_path << std::setw(5) << std::setfill('0') << shift_ix << ".csv";
  save_to_csv(spix, filename, height, width);

  // -- update --
  shift_ix++;
}


// Save all frames in split_prop
template <typename T>
void Logger::save_seq(const std::string& directory, const thrust::device_vector<T>& seq) {

    // int total_size = seq.size();
    // int nimgs = total_size / npix;

    // if (total_size % npix != 0) {
    //     throw std::runtime_error("Vector size is not a multiple of frame size.");
    // }

    // // Create directory if it does not exist
    // if (access(directory.c_str(), F_OK) == -1) {
    //     if (mkdir(directory.c_str(), 0755) != 0) {
    //         throw std::runtime_error("Failed to create directory: " + directory);
    //     }
    // }

    // // // Transfer data to host
    // // thrust::host_vector<int> host_vec(seq);

    // // Save each frame
    // for (int idx = 0; idx < nimgs; ++idx) {
    //     // -- offset by index --
    //     auto img_start = seq.begin() + idx * npix;
    //     auto img_end = img_start + npix;
    //     thrust::device_vector<T> img_data(img_start, img_end);

    //     // -- save --
    //     std::ostringstream filename;
    //     filename << directory << std::setw(5) << std::setfill('0') << idx << ".csv";
    //     save_to_csv(img_data, filename.str(), height, width);
    // }

}


// Save all frames in split_prop
template <typename T>
void Logger::save_seq_frame(const std::string& directory,
                            const thrust::device_vector<T>& img, int seq_index, int height, int width) {

  // // Create directory if it does not exist
  // if (access(directory.c_str(), F_OK) == -1) {
  //   if (mkdir(directory.c_str(), 0755) != 0) {
  //     throw std::runtime_error("Failed to create directory: " + directory);
  //   }
  // }

  // // Set Frame Directory 
  // std::ostringstream _dir0;
  // _dir0 << directory << std::setw(5) << std::setfill('0') << frame_index << "/";
  // const std::string& dir0 = _dir0.str();
  // if (access(directory.c_str(), F_OK) == -1) {
  //   if (mkdir(directory.c_str(), 0755) != 0) {
  //     throw std::runtime_error("Failed to create directory: " + dir0);
  //   }
  // }

  // // Set Filename
  // std::ostringstream filename;
  // filename << dir0 << std::setw(5) << std::setfill('0') << seq_index << ".csv";

  // Get Filename
  auto filename = get_filename(directory,seq_index);

  // Save Image
  save_to_csv(img, filename, height, width);

}


// Save all frames in split_prop
template <typename T>
void Logger::save_seq_v2(const std::string& directory, const thrust::device_vector<T>& seq, int nftrs) {

    int total_size = seq.size();
    int nimgs = total_size / (nftrs*nspix);
    // printf("nimgs,total_size,nftrs,nspix: %d,%d,%d,%d\n",nimgs,total_size,nftrs,nspix);

    if (total_size % (nftrs*nspix) != 0) {
        throw std::runtime_error("Vector size is not a multiple of frame size.");
    }

    // Create directory if it does not exist
    if (access(directory.c_str(), F_OK) == -1) {
        if (mkdir(directory.c_str(), 0755) != 0) {
            throw std::runtime_error("Failed to create directory: " + directory);
        }
    }

    // Transfer data to host
    // thrust::host_vector<int> host_vec(seq);

    // Save each frame
    for (int idx = 0; idx < nimgs; ++idx) {
        // -- offset by index --
        auto img_start = seq.begin() + idx * nspix * nftrs;
        auto img_end = img_start + nspix * nftrs;
        thrust::device_vector<T> img_data(img_start, img_end);

        // -- save --
        std::ostringstream filename;
        filename << directory << std::setw(5) << std::setfill('0') << idx << ".csv";
        save_to_csv(img_data, filename.str(), nspix, nftrs);
    }
}


// Function to save a device vector to CSV
template <typename T>
void Logger::save_to_csv(const thrust::device_vector<T>& device_vec,
                         const std::string& filename, int height, int width) {
    if (device_vec.size() < height * width) {
        throw std::runtime_error("Device vector size is smaller than required dimensions.");
    }

    // Transfer data to host
    thrust::host_vector<T> host_vec(device_vec);

    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write CSV format
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << host_vec[i * width + j];
            if (j < width - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Saved to " << filename << " successfully.\n";
}



// Get filename from directory information
std::string Logger::get_filename(const std::string& directory, int seq_index){

  // Create directory if it does not exist
  if (access(directory.c_str(), F_OK) == -1) {
    if (mkdir(directory.c_str(), 0755) != 0) {
      throw std::runtime_error("Failed to create directory: " + directory);
    }
  }

  // Set Frame Directory 
  std::ostringstream _dir0;
  _dir0 << directory << std::setw(5) << std::setfill('0') << frame_index << "/";
  const std::string& dir0 = _dir0.str();
  if (access(dir0.c_str(), F_OK) == -1) {
    if (mkdir(dir0.c_str(), 0755) != 0) {
      throw std::runtime_error("Failed to create directory: " + dir0);
    }
  }

  // Set Filename
  std::ostringstream filename;
  filename << dir0 << std::setw(5) << std::setfill('0') << seq_index << ".csv";
  return filename.str();
}
