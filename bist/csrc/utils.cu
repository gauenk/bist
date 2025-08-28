
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <type_traits>


// using namespace cv;
using namespace std;


#include "utils.h"



// Set Configuration
// superpixel_options get_sp_options(int nPixels_in_square_side,
//                                          float i_std,float beta, float alpha_hasting){
//     superpixel_options opt;
//     opt.nPixels_in_square_side = nPixels_in_square_side;
//     opt.i_std = i_std;
//     opt.beta_potts_term = beta;
//     opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;
//     opt.s_std = opt.nPixels_in_square_side;
//     opt.prior_count = opt.area*opt.area ;
//     opt.calc_cov = true;
//     opt.use_hex = false;
//     opt.alpha_hasting = alpha_hasting;

//     opt.nEMIters = opt.nPixels_in_square_side;
//     // opt.nEMIters = 10000;
//     //opt.nEMIters = 15;
//     opt.nInnerIters = 4;
//     return opt;
// }


// void save_params(String fname, superpixel_params* sparams, std::vector<int> spix_ids){
//   cout << "\n" << fname << endl;
//     std::ofstream outfile(fname); // Create an ofstream object to write to a file named "output.txt"
//     if (!outfile.is_open()) { // Check if the file opened successfully
//       std::cerr << "Error opening file." << std::endl;
//       exit(1);
//     }
//     int nspix = spix_ids.size();
//     // std::set<int> uniqueSet(myVector.begin(), myVector.end());

//     outfile << "label,mu_i.x,mu_i.y,mu_i.z,mu_s.x,mu_s.y,sigma_s.x,sigma_s.y,sigma_s.z,count,prior_count" << std::endl; // Write to the file
//     // for(int idx=spix_ids.begin(); idx < spix_ids.end(); idx++){
//     for(int spix_index : spix_ids){
//       //int spix_index = spix_ids[idx];
//       superpixel_params sparam = sparams[spix_index];
//       float3 mu_i = sparam.mu_i;
//       double2 mu_s = sparam.mu_s;
//       double3 sigma_s = sparam.sigma_s;

//       // Write the data in CSV format
//       outfile << spix_index << "," << mu_i.x << "," << mu_i.y << "," << mu_i.z << ","
//               << mu_s.x << "," << mu_s.y << ","
//               << sigma_s.x << "," << sigma_s.y << "," << sigma_s.z << ","
//               << sparam.count << "," << sparam.prior_count
//               << std::endl;
//     }
//     outfile.close(); // Close the file
// }


// void show_usage(const std::string &program_name) {
//     std::cout << "Usage: " << program_name << " [options]\n"
//               << "Options:\n"
//               << "  -h, --help                Show this help message\n"
//               << "  -d, --image_direc DIR     Set image directory (default: image/)\n"
//               << "  -n, --nPixels_on_side N   Set number of pixels on side (default: 15)\n"
//               << "  --i_std VALUE             Set i_std value (default: 0.018)\n"
//               << "  --im_size VALUE           Set image size (default: 0)\n"
//               << "  --beta VALUE              Set beta value (default: 0.5)\n"
//               << "  --alpha VALUE             Set alpha value (default: 0.5)\n"
//               << "  --subdir DIR              Set subdirectory (default: none)\n";
// }

