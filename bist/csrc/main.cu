
// -- basic --
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h> // For access()


// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- opencv --
#include <opencv2/opencv.hpp>

// -- local --
#include "file_io.h"
#include "structs.h"
#include "init_utils.h"
#include "rgb2lab.h"
#include "bass.h"
#include "bist.h"
#include "shift_and_fill.h"
#include "seg_utils.h"
#include "split_disconnected.h"
#include "lots_of_merge.h"
#include "main_utils.h"

using namespace std;



int main(int argc, char **argv) {

    // -- init --
    const char *direc = "image/";
    const char *fdirec = "";
    const char *odirec = "./result/";
    const char *img_ext = "jpg";
    string subdir = "";
    int sp_size = 25; // number of pixels along an axis
    int im_size = 0;
    float alpha = log(0.5);
    float sigma_app = 0.009f;
    float potts = 10.0;
    bool read_video = true;
    float gamma = 4.0;
    int input_niters = 0;
    int vid_niters = 0;
    bool overlap_sm = true;
    bool prop_nc = true;
    bool prop_icov = true;
    float epsilon_reid = 1e-6;
    float epsilon_new = 0.05;
    float merge_offset = 0.0;
    float split_alpha = 0.0;
    int target_nspix = 0;
    int logging = 0;
    int nimgs = 0;
    int save_only_spix = 0;
    int _batch_mode = 0;
    int _use_sm = 1;

    /******************************

         -- parse arguments --

    *******************************/

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        }

        if (!parse_argument(i, argc, argv, arg, "-d", direc) ||
            !parse_argument(i, argc, argv, arg, "-f", fdirec) ||
            !parse_argument(i, argc, argv, arg, "-o", odirec) ||
            !parse_argument(i, argc, argv, arg, "-n", sp_size) ||
            !parse_argument(i, argc, argv, arg, "--sigma_app", sigma_app) ||
            !parse_argument(i, argc, argv, arg, "--im_size", im_size) ||
            !parse_argument(i, argc, argv, arg, "--potts", potts) ||
            !parse_argument(i, argc, argv, arg, "--alpha", alpha) ||
            !parse_argument(i, argc, argv, arg, "--split_alpha", split_alpha)||
            !parse_argument(i, argc, argv, arg, "--img_ext", img_ext) ||
            !parse_argument(i, argc, argv, arg, "--read_video", read_video) ||
            !parse_argument(i, argc, argv, arg, "--subdir", subdir) ||
            !parse_argument(i, argc, argv, arg, "--niters", input_niters) ||
            !parse_argument(i, argc, argv, arg, "--use_sm", _use_sm) ||
            !parse_argument(i, argc, argv, arg, "--vid_niters", vid_niters) ||
            !parse_argument(i, argc, argv, arg, "--tgt_nspix", target_nspix) ||
            !parse_argument(i,argc,argv,arg,"--epsilon_reid",epsilon_reid) ||
            !parse_argument(i, argc, argv, arg, "--epsilon_new", epsilon_new) ||
            !parse_argument(i, argc, argv, arg, "--prop_nc", prop_nc) ||
            !parse_argument(i, argc, argv, arg, "--prop_icov", prop_icov) ||
            !parse_argument(i, argc, argv, arg, "--overlap", overlap_sm) ||
            !parse_argument(i, argc, argv, arg, "--gamma", gamma) ||
            !parse_argument(i, argc, argv, arg, "--logging", logging) ||
            !parse_argument(i, argc, argv, arg, "--save_only_spix", save_only_spix) ||
            !parse_argument(i, argc, argv, arg, "--nimgs", nimgs) ||
            !parse_argument(i, argc, argv, arg, "--batch_mode", _batch_mode)) {
            return 1;
        }

        // if (arg == "-n") {
        //     if (sp_size < 1) {
        //         std::cerr << "--sp_size option requires sp_size >= 1." << std::endl;
        //         return 1;
        //     }
        // }
    }

    /******************************

             -- config --

    *******************************/

    DIR *dpdf;
    struct dirent *epdf;

    // -- control the number of spix --
    int nbatch = 1;
    int nftrs = 3;
    bool batch_mode = (_batch_mode == 1);
    bool use_sm = (_use_sm == 1);
    bool controlled_nspix = (target_nspix > 0);
    int niters = (input_niters == 0) ? sp_size : input_niters;
    int niters_seg = 4;
    float sigma2_app = sigma_app*sigma_app;
    int sm_start = -1;
    float sigma2_size = 0.01;
    std::string full_path = string(odirec)+subdir;
    ensureDirectoryExists(string(odirec));
    ensureDirectoryExists(full_path);
    std::vector<string> _img_files = get_image_files(direc, img_ext, true); // always read in order
    if (nimgs == 0){ nimgs = (int)_img_files.size(); }
    std::vector<string> img_files(_img_files.begin(), _img_files.begin() + std::min(nimgs, (int)_img_files.size()));
    printf("num files: %d\n",img_files.size());
    std::vector<std::vector<float>> flows = read_flows(fdirec, img_files.size());
    // printf("use_sm: %d\n",use_sm==true);  

    // -- loop init --
    float niters_ave = 0;
    int count = 0;
    double timer=0;
    int* spix_prev = nullptr;
    SuperpixelParams* params_prev = nullptr;

    for(std::string img_name : img_files){


            /**********************************************

                    Part A: Set-up & Reading

            **********************************************/

            // -- read image --
            cv::String filename = string(direc) + img_name;
            std::cout << "Filename: " << filename << std::endl;
            cv::String img_number =  img_name.substr (0, img_name.find("."));
            cv::Mat image1 = imread(filename, cv::IMREAD_COLOR);
            if(! image1.data ) continue;
            cv::Mat image;
            if (im_size==0)
            {
                image = image1;
            }
            else
            {
                resize(image1, image, cv::Size(im_size,im_size));
            }
            uint8_t* img_raw = image.data;
            cudaDeviceSynchronize();

            // -- read flow --
            float* flow = nullptr;
            if ((flows.size() > 0) && (count>0)){
              flow = flows[count-1].data();
            }
            if (read_video && (count>0)){
              assert(flows.size()>0);
            }

            // -- unpack dims --
            int height = image.rows;
            int width = image.cols;
            int npix = height*width;

            // -- update sp_size to control # of spix --
            if (controlled_nspix){
              float _sp_size = (1.0*height*width) / (1.0*target_nspix);
              sp_size = round(sqrt(_sp_size));
              sp_size = max(sp_size,5);
            }

            /**********************************************

                    Part B: Superpixel Segmentation

            **********************************************/

            // -- start timer --
            clock_t start,finish;
            cudaDeviceSynchronize();
            start = clock();

            // -- prepare images --
            float* img_rgb = rescale_image(img_raw,nbatch,npix,255.);
            float* img_lab = (float*)easy_allocate(nbatch*npix*3,sizeof(float));
            rgb2lab(img_rgb,img_lab,nbatch,npix); // convert image to LAB

            // -- single image --
            int* spix = nullptr;
            bool* border = nullptr;
            int nspix = -1;
            SuperpixelParams* params = nullptr;
            cv::String log_root = string(odirec)+"log/";

            Logger* logger = nullptr;
            if (logging==1){
              ensureDirectoryExists(log_root);
              logger = new Logger(log_root,count,height,width,10*0,niters,4);
            }

            if ((count == 0)||(read_video == false)){
              // sm_start = 3;

              if (batch_mode){
                auto out = run_batched_bass(img_lab, nbatch, height, width, nftrs,
                    niters, niters_seg, sm_start,
                    sp_size, sigma2_app, sigma2_size,
                    potts,alpha,split_alpha,target_nspix,logger);
                spix = std::get<0>(out);
                border = std::get<1>(out);
                params = std::get<2>(out);

              }else{
                auto out = run_bass(img_lab, nbatch, height, width, nftrs,
                    niters, niters_seg, sm_start,
                    sp_size, sigma2_app, sigma2_size,
                    potts,alpha,split_alpha,target_nspix,use_sm,logger);
                spix = std::get<0>(out);
                border = std::get<1>(out);
                params = std::get<2>(out);
              }

              // auto out = run_bass(img_lab, nbatch, height, width, nftrs,
              //                     niters, niters_seg, sm_start,
              //                     sp_size, sigma2_app, sigma2_size,
              //                     potts,alpha,split_alpha,target_nspix,use_sm,logger);
              // spix = std::get<0>(out);
              // border = std::get<1>(out);
              // params = std::get<2>(out);

              // int nspix = params->ids.size();
              // run_invalidate_disconnected(spix, 1, height, width, nspix)
            }else{

              auto out_saf = shift_and_fill(spix_prev,params_prev,flow,
                                            nbatch,height,width,sp_size,
                                            prop_nc,overlap_sm,logger);
              int* filled_spix = std::get<0>(out_saf);
              int* shifted_spix = std::get<1>(out_saf);

              // -- count percentage invalid --
              int ninvalid = count_invalid(shifted_spix,npix);
              float iperc = ninvalid / (1.0*npix);

              // -- pick the number of iterations --
              if (vid_niters > 0){
                niters = vid_niters;
              }else{
                if (iperc > 0.20){
                  niters = 12;
                }else if(iperc < 0.01){
                  niters = 4;
                }else {
                  niters = 8;
                }
              }
              // niters = 12;

              if (logging==1){
                niters = 8; // just for easy animations
              }


              /**********************************

                        -- BIST --

              ***********************************/

              auto out = run_bist(img_lab, nbatch, height, width, nftrs,
                                  niters, niters_seg, sm_start,
                                  sp_size,sigma2_app,sigma2_size,
                                  potts,alpha,filled_spix,shifted_spix,params_prev,
                                  epsilon_reid, epsilon_new,
                                  merge_offset, split_alpha,
                                  gamma, target_nspix, prop_icov,logger);
              spix = std::get<0>(out);
              border = std::get<1>(out);
              params = std::get<2>(out);

              // -- free --
              cudaFree(filled_spix);
              cudaFree(shifted_spix);

            }
            niters_ave += niters;

            // -- end timer --
            cudaDeviceSynchronize();
            finish = clock();
            cout<< "Segmentation takes " <<
              ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;
            timer += (double)(finish - start)/CLOCKS_PER_SEC;

            //  -- error check --
            cudaError_t err_t = cudaDeviceSynchronize();
            if (err_t){
                std::cerr << "CUDA error after cudaDeviceSynchronize."
                          << err_t << std::endl;
                return 0;
            }

            // -- inc counter --
            count++;
            if (logger!=nullptr){ delete logger; }

            /**********************************************
                   Part 2.5: Optionally run "lots of merge"

             *********************************************/

            // nspix = params->ids.size();
            // int _sp_size = 200;
            // auto _alpha = alpha;// - 100.0;
            // auto _out = run_lots_of_merge(img_lab, spix, nspix,
            //                               nbatch, height, width, nftrs,
            //                               niters, niters_seg, sm_start,
            //                               _sp_size,sigma2_app,sigma2_size,
            //                               potts,_alpha,split_alpha);
            // int* _spix = std::get<0>(_out);
            // bool* _border = std::get<1>(_out);
            // SuperpixelParams* _params = std::get<2>(_out);

            /**********************************************

                   Part 2.5.5: Square stuff! [viz only]

            **********************************************/

            // // -- square border --
            // int sq_sp_size = 25;
            // auto _sq_out = get_square_segmentation(sq_sp_size, nbatch, height, width);
            // int* sq_spix = std::get<0>(_sq_out);
            // int sq_nspix = std::get<1>(_sq_out);
            // bool* sq_border = std::get<2>(_sq_out);
            // printf("sq_nspix: %d\n",sq_nspix);

            // // -- save pooled --
            // auto sq_out = get_img_pooled(img_rgb, sq_spix, height, width, sq_nspix);
            // cv::Mat sq_pooled_img = std::get<0>(sq_out);
            // float* sq_pooled_img_ptr = std::get<1>(sq_out);
            // cv::String fname_sq_pooled=string(odirec)+subdir+\
            //   "sq_pooled_"+img_number+".png";
            // imwrite(fname_sq_pooled, sq_pooled_img);

            // // -- save border on pooled image --
            // cv::Mat sq_border_img = get_img_border(sq_pooled_img_ptr, sq_border,
            //                                      height, width, nftrs);
            // cv::String fname_sq_border = string(odirec)+subdir+\
            //   "sq_border_"+img_number+".png";
            // imwrite(fname_sq_border, sq_border_img);
            // cudaFree(sq_pooled_img_ptr);
            // cudaFree(sq_spix);
            // cudaFree(sq_border);


            /**********************************************

                       Part 3: Save Information

            **********************************************/

            // -- save segmentation --
            cv::String seg_idx_path = string(odirec)+subdir+img_number+".csv";
            save_spix_gpu(seg_idx_path, spix, height, width);

            // -- save pooled --
            // nspix = params->ids.size();
            // auto out = get_img_pooled(img_rgb, spix, height, width, nspix);
            // cv::Mat pooled_img = std::get<0>(out);
            // float* pooled_img_ptr = std::get<1>(out);
            // cv::String fname_res_pooled=string(odirec)+subdir+"pooled_"+img_number+".png";
            // if (not save_only_spix){
            //   imwrite(fname_res_pooled, pooled_img);
            // }

            // -- residual image --
            // cv::Mat res_img = get_img_res(img_rgb, pooled_img_ptr, height, width);
            // cv::String fname_res=string(odirec)+subdir+"res_"+img_number+".png";
            // if (not save_only_spix){
            //   imwrite(fname_res, res_img);
            // }

            // -- save border on pooled image --
            // cv::Mat pborder_img = get_img_border(pooled_img_ptr, _border,
            //                                      height, width, nftrs);
            //cv::String fname_pborder=string(odirec)+subdir+"pborder_"+img_number+".png";
            // imwrite(fname_pborder, pborder_img);
            // cudaFree(pooled_img_ptr);

            // -- save border --
            cv::Mat border_img = get_img_border(img_rgb, border, height, width, nftrs);
            cv::String fname_res_border=string(odirec)+subdir+"border_"+img_number+".png";
            if (not save_only_spix){
              imwrite(fname_res_border, border_img);
            }
            // free(border_cpu);
            cudaFree(border);

            // -- save parameters --
            cv::String params_idx_path =string(odirec)+subdir+img_number+"_params.csv";
            // if (not save_only_spix){
            //     save_params(params_idx_path,params);
            // }

            // -- handle memory for previous info --
            if (count>0){
              cudaFree(spix_prev);
              if (params_prev != nullptr){
                delete params_prev;
              } 
            }

            if (count == img_files.size()){
              cudaFree(spix);
              if (params != nullptr){
                delete params;
              } 
            }else{
              spix_prev = spix;
              params_prev = params;
            }


            // -- free images --
            cudaFree(img_lab);
            cudaFree(img_rgb);
            // cudaDeviceReset();

        }
        cudaDeviceReset();

        // -- info --
        if (count > 0){
          cout << "Mean Time: " << timer/count << " ";
          cout << "Mean Iters: " << niters_ave/(1.0*count) << endl;
        }else{
          cout << "no images!" << endl;
        }
}

