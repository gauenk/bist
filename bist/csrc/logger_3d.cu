


// -- nice cumsum --
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
// #include "init_utils.h"


#include "structs_3d.h"
#include "sparams_io_3d.h"
// #include "poly_from_points.h"
#include "border_edges.h"
#include "scannet_reader.h"
#include "seg_utils_3d.h"
#include "csr_edges.h"

#define THREADS_PER_BLOCK 512


// Constructor to initialize all vectors with the given size


Logger::Logger(const std::filesystem::path& _output_root,
               std::vector<std::filesystem::path> scene_paths)
{
    // printf("scene_paths.size(): %d\n",scene_paths.size());
    //exit(1);
    // -- log directory --
    log_roots.resize(scene_paths.size());
    int bx = 0;
    for (const auto& scene_path : scene_paths) {
        std::string scene_name = scene_path.filename().string();
        std::filesystem::path log_root_b = _output_root / scene_name / "log/";
        if (!std::filesystem::exists(log_root_b)) {
            std::cout << "Creaing output path " + log_root_b.string() << std::endl;
            std::filesystem::create_directories(log_root_b);
        }
        log_roots[bx] = log_root_b;
    }
    // printf("log_roots.size(): %d (%d)\n",log_roots.size(),scene_paths.size());
    // exit(1);
}




void Logger::boundary_update(PointCloudData& data, SuperpixelParams3d& params, spix_params* sp_params) {

    // -- ... --
    thrust::host_vector<uint32_t> csum_nspix = params.csum_nspix;
    thrust::device_vector<bool> border_cpy = params.border;
    bool* border = thrust::raw_pointer_cast(border_cpy.data());
    cudaMemset(border, 0, data.V*sizeof(bool));
    set_border(params.spix_ptr(), border, data.csr_edges,data.csr_eptr,data.V);
    
    // -- prepare superpixel information --
    aos_to_soa(sp_params, params); // from spix_params -> SuperpixelParams3d

    // -- get host vectors for the point-cloud data too / this helps vizualize edges --
    auto ftrs = data.ftrs_host();
    auto pos = data.pos_host();
    auto vptr = data.vptr_host();


    //
    // -- too much work.... explain later... pretty silly...
    //

    // // -- just copy for testing --
    // border_edges.resize(2*data.E);
    // thrust::device_ptr<uint32_t> csr_edges_dptr(data.csr_edges);
    // thrust::copy(csr_edges_dptr,csr_edges_dptr+2*data.E,border_edges.begin());
    // border_ptr.resize(data.E+1);
    // thrust::device_ptr<uint32_t> eptr_dptr(data.csr_eptr);
    // thrust::copy(eptr_dptr,eptr_dptr+data.E+1,border_ptr.begin());

    // // -- from csr_edges to pairs [we happen to write this way...] --
    // uint32_t* border_edges_ptr = thrust::raw_pointer_cast(border_edges.data());
    // uint32_t* border_ptr_dptr = thrust::raw_pointer_cast(border_ptr.data());
    // uint32_t* edges;
    // int* eptr;
    // std::tie(edges,eptr) = get_edges_from_csr(border_edges_ptr,border_ptr_dptr,data.ptr,data.bids,data.V,data.B);

    // uint32_t* csr_edges = thrust::raw_pointer_cast(border_edges.data());
    // uint32_t* csr_eptr = thrust::raw_pointer_cast(border_ptr.data());

    uint32_t* edges;
    int* eptr;
    std::tie(edges,eptr) = get_edges_from_csr(data.csr_edges,data.csr_eptr,data.ptr,data.bids,data.V,data.B);


    // -- copy edges to cpu --
    // thrust::host_vector<uint32_t> edges_cpu;
    thrust::host_vector<int> eptr_cpu(data.B+1);

    thrust::device_ptr<int> eptr_dev_ptr(eptr);
    thrust::copy(eptr_dev_ptr,eptr_dev_ptr+data.B+1,eptr_cpu.begin());
    int num_edge_pairs_og = eptr_cpu.back();

    // edges_cpu.resize(2*num_edge_pairs);
    // thrust::device_ptr<uint32_t> edges_dev_ptr(edges);
    // thrust::copy(edges_dev_ptr,edges_dev_ptr+2*num_edge_pairs,edges_cpu.begin());

    //printf("num_edge_pairs_og: %d\n",num_edge_pairs_og);
    // -- get polygon for cooleo scene info points --
    thrust::device_vector<uint32_t> border_edges;
    thrust::device_vector<uint32_t> _tmp;
    std::tie(border_edges,_tmp) = get_border_edges(params.spix_ptr(),border,edges,num_edge_pairs_og);
    //printf("border_edges.size(): %d\n",border_edges.size());
    // thrust::host_vector<uint32_t> edges_cpu = border_edges;
    // get_border_edges(uint32_t* spix, bool* border, uint32_t* edges, uint32_t E);
    int num_edges = border_edges.size()/2;
    thrust::host_vector<uint32_t> edges_cpu = border_edges;

    cudaFree(edges);
    cudaFree(eptr);


    int bx = 0;
    // printf("log_roots.size(): %d\n",log_roots.size());
    for (const auto& log_root_s : log_roots) {

        // -- get save root --
        std::filesystem::path bndy_root = log_root_s / "bndy/";
        if (!std::filesystem::exists(bndy_root)) {
            std::cout << "Creaing output path " + bndy_root.string() << std::endl;
            std::filesystem::create_directories(bndy_root);
        }

        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );


        // -- get save filename --
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << bndy_ix << "_spix.ply"; // "%05d.ply" % bndy_ix
        std::filesystem::path ply_file = bndy_root / ss.str();

        // -- get batch slice --
        int start_idx = csum_nspix[bx];
        int end_idx = csum_nspix[bx + 1];
        int nspix = end_idx - start_idx;
        thrust::host_vector<float3> mu_app(params.mu_app.begin() + start_idx,
                                            params.mu_app.begin() + end_idx);
        thrust::host_vector<double3> mu_pos(params.mu_pos.begin() + start_idx,
                                            params.mu_pos.begin() + end_idx);
        thrust::host_vector<double3> var_pos(params.var_pos.begin() + start_idx,
                                            params.var_pos.begin() + end_idx);
        thrust::host_vector<double3> cov_pos(params.cov_pos.begin() + start_idx,
                                            params.cov_pos.begin() + end_idx);

        // -- write superpixel information --
        ScanNetScene scene;
        bool tmp = scene.write_spix_ply(ply_file,mu_app,mu_pos,var_pos,cov_pos,nspix);


       // -- save boundary edges --
        std::stringstream ss_scene;
        ss_scene << std::setfill('0') << std::setw(5) << bndy_ix << "_edges.ply";  // "%05d.ply" % bndy_ix
        std::filesystem::path scene_fn = bndy_root / ss_scene.str();

        // -- get scene information --
        thrust::host_vector<float> ftrs_b(ftrs.begin() + 3*vptr[bx], ftrs.begin() + 3*vptr[bx+1]);
        thrust::host_vector<float> pos_b(pos.begin() + 3*vptr[bx], pos.begin() +  3*vptr[bx+1]);
        thrust::host_vector<uint32_t> spix_b(params.spix.begin() + vptr[bx], params.spix.begin() +  vptr[bx+1]);
        thrust::host_vector<uint32_t> edges_b = edges_cpu;//(edges_cpu.begin() + 2*eptr_cpu[bx], edges_cpu.begin() +  2*eptr_cpu[bx+1]);
        int Vb = vptr[bx+1] - vptr[bx];
        int Eb = num_edges;//eptr_cpu[bx+1] - eptr_cpu[bx];
        // printf("border_edges.size(): %d %d\n",border_edges.size(),Eb);

        //-- write --
        float* ftrs_b_ptr = thrust::raw_pointer_cast(ftrs_b.data());
        float* pos_b_ptr = thrust::raw_pointer_cast(pos_b.data());
        uint32_t* spix_b_ptr = thrust::raw_pointer_cast(spix_b.data());
        uint32_t* edges_b_ptr = thrust::raw_pointer_cast(edges_b.data());
        tmp = scene.write_ply(scene_fn,ftrs_b_ptr,pos_b_ptr,edges_b_ptr,Vb,Eb);

        bx++;

    }

    // -- increment counter --
    bndy_ix++;

}