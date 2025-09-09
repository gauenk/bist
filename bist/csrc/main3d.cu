
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
//#include <opencv2/opencv.hpp>

// -- local --
#include "init_utils.h"
#include "bass3d.h"
#include "structs_3d.h"
//#include "pointcloud_reader.h"
#include "scannet_reader.h"
#include "csr_edges.h"
#include "graph_coloring.h"
#include "face_dual.h"

#include "manifold_edges.h"
#include "border_edges.h"
// #include "seg_utils_3d.h"

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

    // -- mesh config params --
    int _use_dual = 1;
    
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
    float data_scale = 0.01; // scannetv2

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
            // -- mesh modification parameters --
            !parse_argument(i, argc, argv, arg, "--dual", _use_dual) ||
            // -- core superpixel parameters --
            !parse_argument(i, argc, argv, arg, "-c", data_scale) ||
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
    bool use_dual = (_use_dual == 1);
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
    SpixMetaData args{niters, niters_seg, sm_start, data_scale, sp_size,
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
            std::vector<std::filesystem::path> scene_files_b(scene_files.begin() + s_index, scene_files.begin() + e_index);
            //auto read_out = read_scene(scene_files_b);
            PointCloudData data  = read_scene(scene_files_b);

            // -- enforce manifold edges --
            if (use_dual){
                manifold_edges(data);
            }

            // -- inspect --
            // {
            //     for(int index=data.E-132-4; index < data.E-132+4; index++){
            //         int e0 = data.edges[2*index+0];
            //         int e1 = data.edges[2*index+1];
            //         int bid = data.edge_batch_ids[index];
            //         printf("%d %d %d\n",e0,e1,bid);
            //     }
            //     int E_ptr = data.eptr[1];
            //     printf("E vs E : %d %d\n",data.E,E_ptr);

            //     for(int index=data.V-132-4; index < data.V-132+4; index++){
            //         int bid = data.vertex_batch_ids[index];
            //         printf(" %d\n",bid);
            //     }
            //     int V_ptr = data.vptr[1];
            //     printf("V vs V : %d %d\n",data.V,V_ptr);

            // }

            // -- convert to csr --
            thrust::device_vector<uint32_t> csr_edges;
            thrust::device_vector<uint32_t> csr_eptr;
            std::tie(csr_edges,csr_eptr) = get_csr_graph_from_edges(data.edges_ptr(),data.edge_batch_ids_ptr(),data.eptr_ptr(),data.vptr_ptr(),data.V,data.E);
            data.csr_edges = std::move(csr_edges);
            data.csr_eptr = std::move(csr_eptr);

            // -- check csr_edges --
            // thrust::device_vector<uint32_t> _edges_chk;
            // thrust::device_vector<int> _eptr_chk;
            // std::tie(_edges_chk,_eptr_chk) = get_edges_from_csr(data.csr_edges_ptr(),data.csr_eptr_ptr(),data.vptr_ptr(),data.vertex_batch_ids_ptr(),data.V,data.B);

            // -- get graph colors --
            uint8_t gchrome;
            thrust::device_vector<uint8_t> gcolors;
            std::tie(gcolors,gchrome) = get_graph_coloring(data.csr_edges_ptr(), data.csr_eptr_ptr(), data.V);
            //printf("graph chromaticity: %d\n",gchrome);
            data.gcolors = std::move(gcolors);
            data.gchrome = gchrome;

            // -- get face-dual mesh --
            PointCloudData dual = use_dual ? create_dual_mesh(data) : data.copy();
            if(use_dual){            
                uint8_t gchrome;
                thrust::device_vector<uint8_t> gcolors;
                std::tie(gcolors,gchrome) = get_graph_coloring(dual.csr_edges_ptr(), dual.csr_eptr_ptr(), dual.V);
                //printf("graph chromaticity: %d\n",gchrome);
                dual.gcolors = std::move(gcolors);
                dual.gchrome = gchrome;
            }


            /**********************************************

                    Part B: Superpixel Segmentation

            **********************************************/

            // -- create logger --
            Logger* logger = nullptr;
            if (logging==1){
                logger = new Logger(output_root,scene_files_b);
            }
            
            // -- start timer --
            clock_t start,finish;
            cudaDeviceSynchronize();
            start = clock();

            // -- run segmentation --
            SuperpixelParams3d params = run_bass3d(dual, args, logger);
            //data.labels = params.spix;
            dual.labels = params.spix;

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

                       Part 3: Save Information

            **********************************************/

            // -- keep edges only --
            //dual.F = 0;
            // PointCloudData border_data = dual.copy();
            // border_data.labels = params.spix;
            // border_data.border = params.border;
            // // cudaMemset(border_data.border_ptr(), 0, border_data.V*sizeof(bool));
            // // set_border(border_data.labels_ptr(), border_data.border_ptr(), border_data.csr_edges_ptr(),border_data.csr_eptr_ptr(),border_data.V);
            // filter_to_border_edges(border_data);
            PointCloudData border_data = get_border_data(data,dual,params,params.nspix_sum,use_dual);


            // -- write and free --
            //bool succ = write_scene(scene_files_b,output_root,data);
            //succ = write_spix(scene_files_b,output_root,params);
            cudaDeviceSynchronize();

            // -- write labeled mesh, spix, and dual mesh --
            for (int batch_index=0; batch_index < data.B; batch_index++){
                
                // init scene [not really needed...]
                ScanNetScene scene;

                // -- get the save path --
                std::string scene_name = scene_files_b[batch_index].filename().string();
                std::filesystem::path write_path = output_root / scene_name;
                if (!std::filesystem::exists(write_path)) {
                    std::filesystem::create_directories(write_path);
                }
                std::filesystem::path mesh_fname = write_path / (scene_name + "_vh_clean_2.ply");
                std::filesystem::path border_fname = write_path / (scene_name + "_border.ply");
                std::filesystem::path dual_fname = write_path / (scene_name + "_dual.ply");
                std::filesystem::path dual_edge_fname = write_path / (scene_name + "_dual_edges.ply");
                std::filesystem::path spix_fname = write_path / (scene_name + "_spix.ply");
                std::cout << mesh_fname << std::endl;

                // -- get batch data --
                PointCloudDataHost host_data(data,batch_index);
                PointCloudDataHost host_border_data(border_data,batch_index);
                PointCloudDataHost host_dual(dual,batch_index);
                SuperpixelParams3dHost host_spix(params,batch_index);

                // -- write mesh scene --
                if(!scene.write_ply(mesh_fname,host_data)){
                    exit(1);
                }
        
                // -- write border scene --
                if(!scene.write_ply(border_fname,host_border_data,true,false)){
                    exit(1);
                }

                // // -- temp --
                // printf("----------------------------\n");
                // auto faces = dual.faces;
                // auto faces_eptr = dual.faces_eptr;
                // for(int index = 0; index < dual.F; index++){

                //     int start = dual.faces_eptr[index];
                //     int end   = dual.faces_eptr[index+1];
                //     bool any_zero = false;
                //     for(int j_index = start; j_index < end; j_index++){
                //         any_zero = any_zero || (faces[j_index] == 0);
                //     }
                //     if(any_zero){
                //         printf("face[%d]: %d\n",index,end-start);
                //     }
                //     for(int j_index = start; j_index < end; j_index++){
                //         if(any_zero){
                //             uint32_t v = faces[j_index];
                //             printf("%d: %d\n",index,v);
                //         }
                //     }
                // }
                // printf("----------------------------\n");



                // -- write dual scene --
                if(!scene.write_ply(dual_fname,host_dual,true,true)){
                    exit(1);
                }

                // -- write dual scene --
                if(!scene.write_ply(dual_edge_fname,host_dual,true,false)){
                    exit(1);
                }


                // -- write spix scene --
                if(!scene.write_spix_ply(spix_fname,host_spix)){
                    exit(1);
                }


            }


            // -- free graph coloring --
            // cudaFree(gcolors);
            // cudaFree(csr_edges);
            // cudaFree(csr_eptr);

            // -- free dev --
            // cudaFree(_edges_chk);
            // cudaFree(_eptr_chk);

            // -- free data --
            // cudaFree(ftrs);
            // cudaFree(pos);
            // cudaFree(edges);
            // cudaFree(bids);
            // cudaFree(ebids);
            // cudaFree(vptr);
            // cudaFree(eptr);
            // cudaFree(dim_sizes);

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
