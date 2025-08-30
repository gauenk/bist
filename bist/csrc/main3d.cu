
// -- basic --
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h> // For access()
#include <filesystem>

// -- thrust --
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

// -- opencv --
#include <opencv2/opencv.hpp>

// -- local --
#include "init_utils.h"
#include "bass3d.h"
#include "structs_3d.h"
//#include "pointcloud_reader.h"
#include "scannet_reader.h"
#include "csr_edges.h"
#include "graph_coloring.h"

using namespace std;


// -- parser --
template <typename T>
bool parse_argument(int &i, int argc, char **argv, const std::string &arg,
                    const std::string &option, T &value) {
    if (arg == option) {
        if (i + 1 < argc) {
            ++i;
            if constexpr (std::is_same<T, int>::value) {
                value = std::stoi(argv[i]);
            } else if constexpr (std::is_same<T, float>::value) {
                value = std::stof(argv[i]);
            } else if constexpr (std::is_same<T, const char *>::value || std::is_same<T, std::string>::value) {
                value = argv[i];
            }
        } else {
            std::cerr << option << " option requires an argument." << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char **argv) {

    // -- reading file io --
    const char *_read_root = "scenes/";
    const char *_output_root = "./result/";
    const char* data_name = "scannetv2";
    int nscenes = 0;
    int batchsize = 1;

    // -- core superpixel parameters --
    int sp_size = 25;
    float sigma_app = 0.009f;
    float potts = 10.0;

    // -- split/merge parameters --
    float alpha = log(0.5);
    float split_alpha = 0.0;

    // -- video split/merge/relabeling parameters --
    float gamma = 4.0;
    float epsilon_reid = 1e-6;
    float epsilon_new = 0.05;

    // -- misc --
    int input_niters = 0;
    int _use_sm = 1;
    int target_nspix = 0;
    int logging = 0;
    int save_only_spix = 0;
    int nspix_buffer_mult = 1;

    /******************************

         -- parse arguments --

    *******************************/

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            //show_usage(argv[0]);
            printf("halp.\n");
            return 0;
        }

        if (// -- reading file io --
            !parse_argument(i, argc, argv, arg, "-r", _read_root) ||
            !parse_argument(i, argc, argv, arg, "-o", _output_root) ||
            !parse_argument(i, argc, argv, arg, "-d", data_name) ||
            !parse_argument(i, argc, argv, arg, "-n", nscenes) ||
            !parse_argument(i, argc, argv, arg, "-b", batchsize) ||
            // -- core superpixel parameters --
            !parse_argument(i, argc, argv, arg, "-s", sp_size) ||
            !parse_argument(i, argc, argv, arg, "--sigma_app", sigma_app) ||
            !parse_argument(i, argc, argv, arg, "--potts", potts) ||
            // -- split/merge parameters --
            !parse_argument(i, argc, argv, arg, "--alpha", alpha) ||
            !parse_argument(i, argc, argv, arg, "--split_alpha", split_alpha) ||
            // -- video split/merge/relabeling parameters --
            !parse_argument(i, argc, argv, arg, "--gamma", gamma) ||
            !parse_argument(i, argc, argv, arg, "--epsilon_reid",epsilon_reid) ||
            !parse_argument(i, argc, argv, arg, "--epsilon_new", epsilon_new) ||
            // -- misc --
            !parse_argument(i, argc, argv, arg, "--niters", input_niters) ||
            !parse_argument(i, argc, argv, arg, "--use_sm", _use_sm) ||
            !parse_argument(i, argc, argv, arg, "--tgt_nspix", target_nspix) ||
            !parse_argument(i, argc, argv, arg, "--logging", logging) ||
            !parse_argument(i, argc, argv, arg, "--save_only_spix", save_only_spix)) {
            return 1;
        }
    }

    /******************************

             -- config --

    *******************************/

    DIR *dpdf;
    struct dirent *epdf;

    // -- paths --
    std::filesystem::path read_root = _read_root;
    std::filesystem::path output_root = _output_root;
    if (!std::filesystem::exists(read_root)) {
        throw std::runtime_error("Reading path " + read_root.string() + " must exist");
    }
    if (!std::filesystem::exists(output_root)) {
        std::cout << "Creaing output path " + output_root.string() << std::endl;
        std::filesystem::create_directories(output_root);
    }


    // -- control the number of spix --
    int nbatch = 1;
    int nftrs = 3;
    bool use_sm = (_use_sm == 1);
    bool controlled_nspix = (target_nspix > 0);
    int niters = (input_niters == 0) ? sp_size : input_niters;
    int niters_seg = 4;
    float sigma2_app = sigma_app*sigma_app;
    int sm_start = -1;
    float sigma2_size = 0.01;
    std::vector<std::filesystem::path> _scene_files = get_scene_files(read_root);
    if (nscenes <= 0){ nscenes = (int)_scene_files.size(); }
    else{ nscenes = std::min(nscenes, (int)_scene_files.size()); }
    std::vector<std::filesystem::path> scene_files(_scene_files.begin(), _scene_files.begin() + nscenes);
    printf("num files: %d\n",scene_files.size());
    // printf("use_sm: %d\n",use_sm==true);  

    // -- create argument struct --
    SpixMetaData args{niters, niters_seg, sm_start, sp_size,
                     sigma2_app, sigma2_size, potts, alpha, split_alpha, target_nspix, nspix_buffer_mult};

    // -- allow for batching --
    int nbatches = nscenes / batchsize;

    // -- loop init --
    float niters_ave = 0;
    int count = 0;
    double timer=0;
    for (int _ix = 0; _ix < nbatches; _ix++){


            /**********************************************

                    Part A: Set-up & Reading

            **********************************************/

            // -- read batch of scenes --
            int s_index = _ix*batchsize;
            int e_index = std::min((_ix+1)*batchsize, (int)scene_files.size());
            int bsize = e_index - s_index;
            std::vector<std::filesystem::path> scene_files_b(scene_files.begin() + s_index, scene_files.begin() + e_index);
            auto read_out = read_scene(scene_files_b);
            float3* ftrs = std::get<0>(read_out);
            float3* pos = std::get<1>(read_out);
            uint32_t* edges = std::get<2>(read_out);
            uint8_t* bids = std::get<3>(read_out);
            uint8_t* ebids = std::get<4>(read_out);
            int* vptr = std::get<5>(read_out);
            int* eptr = std::get<6>(read_out);
            float* dim_sizes = std::get<7>(read_out);
            int nbatch_b = scene_files_b.size();
            cudaDeviceSynchronize();

            // -- some reading; idk why not just return it by whatever --
            int V_total;
            cudaMemcpy(&V_total,&vptr[nbatch_b],sizeof(int),cudaMemcpyDeviceToHost);
            int E_total;
            cudaMemcpy(&E_total,&eptr[nbatch_b],sizeof(int),cudaMemcpyDeviceToHost);
            printf("V_total E_total : %d %d\n",V_total,E_total);

            // -- convert to csr --
            uint32_t* csr_edges;
            uint32_t* csr_eptr;
            std::tie(csr_edges,csr_eptr) = get_csr_graph_from_edges(edges,ebids,eptr,vptr,V_total,E_total);
            
            // // -- check csr_edges --
            // uint32_t* _edges_chk;
            // int* _eptr_chk;
            // std::tie(_edges_chk,_eptr_chk) = get_edges_from_csr(csr_edges,csr_eptr,vptr,bids,V_total,nbatch_b);

            // -- get graph colors --
            uint8_t* gcolors;
            uint8_t gchrome;
            std::tie(gcolors,gchrome) = get_graph_coloring(csr_edges, csr_eptr, V_total);
            printf("graph chromaticity: %d\n",gchrome);

            // -- update sp_size to control # of spix --
            // if (controlled_nspix){
            //   float _sp_size = (1.0*height*width) / (1.0*target_nspix);
            //   sp_size = round(sqrt(_sp_size));
            //   sp_size = max(sp_size,5);
            // }

            /**********************************************

                    Part B: Superpixel Segmentation

            **********************************************/

            // -- single image --
            // uint32_t* spix = nullptr;
            // bool* border = nullptr;
            // int nspix = -1;
            // SuperpixelParams* params = nullptr;

            // -- start timer --
            clock_t start,finish;
            cudaDeviceSynchronize();
            start = clock();

            
            Logger* logger = nullptr;
            // -- run segmentation --
            PointCloudData data{ftrs, pos, gcolors, csr_edges, bids, 
                                vptr, csr_eptr, dim_sizes, gchrome, 
                                scene_files_b.size(), V_total, E_total};
            if (logging==1){
                logger = new Logger(output_root,scene_files_b);
            }
            SuperpixelParams3d params = run_bass3d(data, args, logger);

            // -- benchmarking/tracking --
            niters_ave += niters;
            cudaDeviceSynchronize();
            finish = clock();
            cout<< "Segmentation takes " <<
              ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;
            timer += (double)(finish - start)/CLOCKS_PER_SEC;
            assert(check_cuda_error()==0);

            // -- inc counter --
            count++;
            if (logger!=nullptr){ delete logger; }

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
            // cv::String fname_sq_pooled=string(output_root)+subdir+\
            //   "sq_pooled_"+img_number+".png";
            // imwrite(fname_sq_pooled, sq_pooled_img);

            // // -- save border on pooled image --
            // cv::Mat sq_border_img = get_img_border(sq_pooled_img_ptr, sq_border,
            //                                      height, width, nftrs);
            // cv::String fname_sq_border = string(output_root)+subdir+\
            //   "sq_border_"+img_number+".png";
            // imwrite(fname_sq_border, sq_border_img);
            // cudaFree(sq_pooled_img_ptr);
            // cudaFree(sq_spix);
            // cudaFree(sq_border);


            /**********************************************

                       Part 3: Save Information

            **********************************************/

            // -- todo! save segmentation --
            // save_seg
            // cv::String seg_idx_path = string(output_root)+img_number+".csv";
            // save_spix_gpu(seg_idx_path, spix, height, width);
            //my_write_function(string(output_root),scene_files,start_index,batchsize);

            // -- save pooled --
            // nspix = params->ids.size();
            // auto out = get_img_pooled(img_rgb, spix, height, width, nspix);
            // cv::Mat pooled_img = std::get<0>(out);
            // float* pooled_img_ptr = std::get<1>(out);
            // cv::String fname_res_pooled=string(output_root)+subdir+"pooled_"+img_number+".png";
            // if (not save_only_spix){
            //   imwrite(fname_res_pooled, pooled_img);
            // }

            // -- residual image --
            // cv::Mat res_img = get_img_res(img_rgb, pooled_img_ptr, height, width);
            // cv::String fname_res=string(output_root)+subdir+"res_"+img_number+".png";
            // if (not save_only_spix){
            //   imwrite(fname_res, res_img);
            // }

            // -- save border on pooled image --
            // cv::Mat pborder_img = get_img_border(pooled_img_ptr, _border,
            //                                      height, width, nftrs);
            //cv::String fname_pborder=string(output_root)+subdir+"pborder_"+img_number+".png";
            // imwrite(fname_pborder, pborder_img);
            // cudaFree(pooled_img_ptr);

            // -- save border --
            // cv::Mat border_img = get_img_border(img_rgb, border, height, width, nftrs);
            // cv::String fname_res_border=string(output_root)+"border_"+img_number+".png";
            // if (not save_only_spix){
            //   imwrite(fname_res_border, border_img);
            // }
            // // free(border_cpu);
            // cudaFree(border);

            // -- save parameters --
            // cv::String params_idx_path =string(output_root)+img_number+"_params.csv";
            // if (not save_only_spix){
            //     save_params(params_idx_path,params);
            // }


            // -- write and free --
            // bool succ = write_scene(scene_files_b,output_root,ftrs,pos,edges,vptr,eptr,spix);
            // bool succ = write_scene(scene_files_b,output_root,ftrs,pos,_edges_chk,vptr,_eptr_chk,gcolors,spix);
            bool succ = write_scene(scene_files_b,output_root,ftrs,pos,edges,vptr,eptr,gcolors,params.spix_ptr());
            succ = write_spix(scene_files_b,output_root,params);
            cudaDeviceSynchronize();

            // -- free graph coloring --
            cudaFree(gcolors);
            cudaFree(csr_edges);
            cudaFree(csr_eptr);

            // -- free dev --
            // cudaFree(_edges_chk);
            // cudaFree(_eptr_chk);

            // -- free data --
            cudaFree(ftrs);
            cudaFree(pos);
            cudaFree(edges);
            cudaFree(bids);
            cudaFree(ebids);
            cudaFree(vptr);
            cudaFree(eptr);
            cudaFree(dim_sizes);

        }
        cudaDeviceReset();

        // -- info --
        if (count > 0){
          cout << "Mean Time: " << timer/count << " ";
          cout << "Mean Iters: " << niters_ave/(1.0*count) << endl;
        }else{
          cout << "no scenes!" << endl;
        }
    return 0;
}
