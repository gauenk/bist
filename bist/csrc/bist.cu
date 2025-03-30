
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
#include "prop.h"
#include "shift_and_fill.h"
#include "seg_utils.h" // dev only
#include "split_disconnected.h"
#include "lots_of_merge.h"

// -- demo --
#include "demo_utils.h"

// using namespace cv;
using namespace std;


int main(int argc, char **argv) {

    // -- init --
    const char *direc = "image/";
    const char *fdirec = "";
    const char *odirec = "../result/";
    const char *img_ext = "jpg";
    string subdir = "";
    int sp_size = 25; // number of pixels along an axis
    int im_size = 0;
    // float i_std = 0.018f;
    // float beta = 0.5f;
    float alpha = log(0.5);
    float sigma_app = 0.009f;
    float potts = 10.0;
    bool read_video = true;
    float iperc_coeff = 4.0;
    int input_niters = 0;
    int vid_niters = 0;
    bool prop_nc = true;
    bool prop_icov = true;

    // float thresh_relabel = 1e-3;
    // float thresh_relabel = 1e-4;
    // float thresh_relabel = 1e-4;
    // float thresh_new = 0.1;
    // float thresh_new = 5e-2;
    // float thresh_new = 5e-2;
    // float thresh_new = 1e-1; // changes the TEX
    // float thresh_new = 1e-5;
    // float thresh_new = 1e-2;
    // float thresh_new = 1e-5;
    // float thresh_new = 5e-4;
    // float merge_offset = 2e-3;
    // float merge_offset = 2e-5;

    // -- new defaults --
    float thresh_relabel = 1e-6;
    float thresh_new = 0.05;
    // float thresh_new = 1e-1;
    // float thresh_new = 1e-1;

    // -- davis params --
    // float thresh_relabel = 1e-5;
    // float thresh_new = 1e-2;

    float merge_offset = 0.0;
    float split_alpha = 0.0;
    int target_nspix = 0;
    int logging = 0;
    int nimgs = 0;

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
            !parse_argument(i, argc, argv, arg, "--image_direc", direc) ||
            !parse_argument(i, argc, argv, arg, "-o", odirec) ||
            !parse_argument(i, argc, argv, arg, "-n", sp_size) ||
            !parse_argument(i,argc,argv,arg,"--nPixels_on_side",sp_size)||
            !parse_argument(i, argc, argv, arg, "--sigma_app", sigma_app) ||
            !parse_argument(i, argc, argv, arg, "--im_size", im_size) ||
            !parse_argument(i, argc, argv, arg, "--potts", potts) ||
            !parse_argument(i, argc, argv, arg, "--alpha", alpha) ||
            !parse_argument(i, argc, argv, arg, "--split_alpha", split_alpha) ||
            !parse_argument(i, argc, argv, arg, "--img_ext", img_ext) ||
            !parse_argument(i, argc, argv, arg, "--read_video", read_video) ||
            !parse_argument(i, argc, argv, arg, "--subdir", subdir) ||
            !parse_argument(i, argc, argv, arg, "--niters", input_niters) ||
            !parse_argument(i, argc, argv, arg, "--vid_niters", vid_niters) ||
            !parse_argument(i, argc, argv, arg, "--tgt_nspix", target_nspix) ||
            !parse_argument(i, argc, argv, arg, "--thresh_relabel", thresh_relabel) ||
            !parse_argument(i, argc, argv, arg, "--thresh_new", thresh_new) ||
            !parse_argument(i, argc, argv, arg, "--prop_nc", prop_nc) ||
            !parse_argument(i, argc, argv, arg, "--prop_icov", prop_icov) ||
            !parse_argument(i, argc, argv, arg, "--iperc_coeff", iperc_coeff) ||
            !parse_argument(i, argc, argv, arg, "--logging", logging) ||
            !parse_argument(i, argc, argv, arg, "--nimgs", nimgs)) {
            return 1;
        }

        if (arg == "-n" || arg == "--nPixels_on_side") {
            if (sp_size < 1) {
                std::cerr << "--sp_size option requires sp_size >= 1." << std::endl;
                return 1;
            }
        }
    }

    // -- control the number of spix --
    bool controlled_nspix = (target_nspix > 0);
    // int target_nspix = 1000;
    printf("prop_nc, prop_icov: %d,%d\n",prop_nc,prop_icov);
    // int target_nspix = 200;
    // int target_nspix = 1400;
    // int target_nspix = 250;
    // int target_nspix = 300;
    // int target_nspix = 300;
    // int target_nspix = 1500;


    /******************************

             -- init spix --

    *******************************/

    DIR *dpdf;
    struct dirent *epdf;

    int nbatch = 1;
    int nftrs = 3;

    // int niters = sp_size;
    int niters = (input_niters == 0) ? sp_size : input_niters;
    // int niters = 10*sp_size;
    // int niters = 100*sp_size;
    // int niters = 200*sp_size;
    int niters_seg = 4;
    float sigma2_app = sigma_app*sigma_app;
    int sm_start = -1;
    float sigma2_size = 0.01;
    // float sigma2_size = 10000.;
    printf("[potts,log alpha] = %2.3f %2.3f\n",potts,alpha);

    std::vector<string> _img_files = get_image_files(direc, img_ext, read_video);
    if (nimgs == 0){ nimgs = (int)_img_files.size(); }
    std::vector<string> img_files(_img_files.begin(), _img_files.begin() + std::min(nimgs, (int)_img_files.size()));
    printf("num files: %d\n",img_files.size());

    // std::vector<string> img_files(_img_files.begin(), _img_files.begin()+10);
    // std::vector<cv::Mat> flows = read_flows(fdirec, img_files.size());
    // std::vector<float*> flows = read_flows(fdirec, img_files.size());
    // int start_index = img_files[0]
    std::vector<std::vector<float>> flows = read_flows(fdirec, img_files.size());
    // std::cout << flows.size() << std::endl;
    // exit(1);

    // -- loop init --
    float niters_ave = 0;
    int count = 0;
    double timer=0;
    int* spix_prev = nullptr;
    SuperpixelParams* params_prev = nullptr;
    // Logger* logger = nullptr;

    for(std::string img_name : img_files){



          // if ((count > 0) and read_video){
          //     sm_start = 0;
          //     // niters = 12;
          //     niters = 8;
          //     // alpha = 0.693;
          //     // alpha = 0.20; // (10.,-3.5,3.5)
          //     // alpha = 0.15; // (1.,-3.5,3.5)
          //     // if (alpha >= 4.0){
          //     //   alpha = 0.1;
          //     // }else{
          //     //   alpha = 0.0;
          //     // }
          //       // niters_seg = 4;
          //   }

            // if (count >= 3){
            //   exit(1);
            // }
            // std::cout << "\n" << std::endl;

            /**********************************************

                    Part 1: Set-up & Reading

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
            // cout << "\nFilename: " << filename <<endl;
            cudaDeviceSynchronize();

            // -- read flow --
            float* flow = nullptr;
            if ((flows.size() > 0) && (count>0)){
              // cv::Mat _flow = flows[count-1];
              // flow = (float*)_flow.data;
              flow = flows[count-1].data();
              // printf("[demo.cu] flow[0,0]: %2.3f,%2.3f\n",flow[0],flow[1]);
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
              // sp_size = min(max(sp_size,5),100);
              sp_size = max(sp_size,5);
            }

            /**********************************************

                    Part 2: Superpixel Segmentation

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
              logger = new Logger(log_root,count,height,width,10*0,niters,4);
            }

            if ((count == 0)||(read_video == false)){
              // sm_start = 3;
              auto out = run_bass(img_lab, nbatch, height, width, nftrs,
                                  niters, niters_seg, sm_start,
                                  sp_size,sigma2_app,sigma2_size,
                                  potts,alpha,split_alpha,target_nspix,logger);
              spix = std::get<0>(out);
              border = std::get<1>(out);
              params = std::get<2>(out);

              // int nspix = params->ids.size();
              // run_invalidate_disconnected(spix, 1, height, width, nspix);
              // int ninvalid = count_invalid(spix,npix);
              // printf("num invalid: %d\n",ninvalid);
              // assert(ninvalid==0);
              // exit(1);

            // }else if ((logging==1) and (count>0)){

            //   niters = 8;
            //   int nspix_prev = params_prev->ids.size();
            //   cv::String log_root = string(odirec)+"log/";
            //   Logger* logger = new Logger(log_root,count,height,width,10*nspix_prev,niters,4);

            //   auto out_saf = shift_and_fill_log(spix_prev,params_prev,flow,
            //                                     nbatch,height,width,prop_nc,logger);
            //   int* filled_spix = std::get<0>(out_saf);
            //   int* shifted_spix = std::get<1>(out_saf);
            //   // int ninvalid = count_invalid(shifted_spix,npix);
            //   // float iperc = ninvalid / (1.0*npix);
            //   auto out = run_prop_log(img_lab, nbatch, height, width, nftrs,
            //                           niters, niters_seg, sm_start,
            //                           sp_size,sigma2_app,sigma2_size,
            //                           potts,alpha,filled_spix,shifted_spix,params_prev,
            //                           thresh_relabel, thresh_new,
            //                           merge_offset, split_alpha,
            //                           iperc_coeff, target_nspix, prop_icov, logger);
            //   spix = std::get<0>(out);
            //   border = std::get<1>(out);
            //   params = std::get<2>(out);

            //   // -- free --
            //   cudaFree(filled_spix);
            //   cudaFree(shifted_spix);


            //   // -- save logger info --
            //   // logger->save_split_prop_frames(log_root);
            //   logger->save();
            //   delete logger;

            }else{

              auto out_saf = shift_and_fill(spix_prev,params_prev,flow,
                                            nbatch,height,width,prop_nc,logger);
              int* filled_spix = std::get<0>(out_saf);
              int* shifted_spix = std::get<1>(out_saf);

              // -- count percentage invalid --
              int ninvalid = count_invalid(shifted_spix,npix);
              float iperc = ninvalid / (1.0*npix);
              // niters = 8;
              // sm_start = -1;

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

                    -- [dev] save border --

              ***********************************/
              // cv::String fname = "shifted_spix.csv";
              // save_spix_gpu(fname, shifted_spix, height, width);
              // cv::String fname1 = "shifted_spix.png";
              // bool* border_tmp = (bool*)easy_allocate(npix,sizeof(bool));
              // CudaFindBorderPixels_end(shifted_spix,border_tmp,npix,nbatch,width,height);
              // cv::Mat border_img = get_img_border(img_rgb,border_tmp,height,width,nftrs);
              // imwrite(fname1,border_img);
              // cudaFree(border_tmp);

              // printf(".\n");
              // cudaMemset(img_lab,0,3*npix*sizeof(float));
              // printf("niters: %d\n",niters);
              auto out = run_prop(img_lab, nbatch, height, width, nftrs,
                                  niters, niters_seg, sm_start,
                                  sp_size,sigma2_app,sigma2_size,
                                  potts,alpha,filled_spix,shifted_spix,params_prev,
                                  thresh_relabel, thresh_new,
                                  merge_offset, split_alpha,
                                  iperc_coeff, target_nspix, prop_icov,logger);
              spix = std::get<0>(out);
              border = std::get<1>(out);
              params = std::get<2>(out);

              // -- free --
              cudaFree(filled_spix);
              cudaFree(shifted_spix);

            }
            niters_ave += niters;

            // // -- after! --
            // cv::String _fname1 = "after_spix.png";
            // bool* _border_tmp = (bool*)easy_allocate(npix,sizeof(bool));
            // CudaFindBorderPixels_end(spix,_border_tmp,npix,nbatch,width,height);
            // cv::Mat _bimg=get_img_border(img_rgb,_border_tmp,height,width,nftrs);
            // // cv::Mat _bimg=get_img_border(img_rgb,border,height,width,nftrs);
            // imwrite(_fname1,_bimg);
            // cudaFree(_border_tmp);

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
            // int* spix_cpu = (int*)easy_allocate_cpu(npix,sizeof(int));
            // cudaMemcpy(spix_cpu, spix, npix * sizeof(int), cudaMemcpyDeviceToHost);
            // save_spix(seg_idx_path,spix_cpu,height,width);
            // free(spix_cpu);
            // cudaFree(spix);

            // bool* border_cpu = (bool*)easy_allocate_cpu(npix,sizeof(bool));
            // cudaMemcpy(border_cpu,border,npix*sizeof(bool), cudaMemcpyDeviceToHost);

            // -- save pooled --
            nspix = params->ids.size();
            auto out = get_img_pooled(img_rgb, spix, height, width, nspix);
            cv::Mat pooled_img = std::get<0>(out);
            float* pooled_img_ptr = std::get<1>(out);
            cv::String fname_res_pooled=string(odirec)+subdir+"pooled_"+img_number+".png";
            imwrite(fname_res_pooled, pooled_img);

            // -- residual image --
            cv::Mat res_img = get_img_res(img_rgb, pooled_img_ptr, height, width);
            cv::String fname_res=string(odirec)+subdir+"res_"+img_number+".png";
            imwrite(fname_res, res_img);

            // -- save border on pooled image --
            // cv::Mat pborder_img = get_img_border(pooled_img_ptr, _border,
            //                                      height, width, nftrs);
            //cv::String fname_pborder=string(odirec)+subdir+"pborder_"+img_number+".png";
            // imwrite(fname_pborder, pborder_img);
            // cudaFree(pooled_img_ptr);

            // -- save border --
            cv::Mat border_img = get_img_border(img_rgb, border, height, width, nftrs);
            cv::String fname_res_border=string(odirec)+subdir+"border_"+img_number+".png";
            imwrite(fname_res_border, border_img);
            // free(border_cpu);
            cudaFree(border);

            // -- save parameters --
            cv::String params_idx_path =string(odirec)+subdir+img_number+"_params.csv";
            save_params(params_idx_path,params);

            // -- save pooled image --
            // Mat mean_img = sp.get_img_cartoon();
            // String fname_res_mean=string(odirec) + subdir + "mean_"+img_number+".png";
            // imwrite(fname_res_mean, mean_img);
            // cout << "saving " << fname_res_mean << endl;

            // -- handle memory for previous info --
            if (count>0){
              cudaFree(spix_prev);
              delete params_prev;
              // free(params_prev); // memcheck didn't catch this error....
            }
            // printf("count,img_files.size(): %d,%d\n",count,img_files.size());
            if (count == img_files.size()){
              // printf("\n\n\n extra free. \n\n");
              cudaFree(spix);
              delete params;
              // free(params); // memcheck didn't catch this error....
              // printf("\n\n\n extra free. \n\n");
            }else{
              spix_prev = spix;
              params_prev = params;
            }

            // spix_prev = spix;
            // params_prev = params;

            // -- free images --
            cudaFree(img_lab);
            cudaFree(img_rgb);
            // cudaDeviceReset();

            // -- free optional --
            // cudaFree(_spix);
            // cudaFree(_border);


            // -- dev --
            // if (count==2){
            // if (count==16){ // frame+1
            //   return 0;
            // }
            // if (count==7){
            //   break;
            // }

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

