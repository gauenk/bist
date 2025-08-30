


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
#include "poly_from_points.h"
#include "scannet_reader.h"
#include "seg_utils_3d.h"

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

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // -- ... --
    thrust::host_vector<uint32_t> csum_nspix = params.csum_nspix;
    thrust::device_vector<bool> border_cpy = params.border;
    bool* border = thrust::raw_pointer_cast(border_cpy.data());
    cudaMemset(border, 0, data.V*sizeof(bool));
    set_border(params.spix_ptr(), border, data.csr_edges,data.csr_eptr,data.V);

    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    int bx = 0;
    // printf("log_roots.size(): %d\n",log_roots.size());
    for (const auto& log_root_s : log_roots) {

        // -- get save root --
        std::filesystem::path bndy_root = log_root_s / "bndy/";
        if (!std::filesystem::exists(bndy_root)) {
            std::cout << "Creaing output path " + bndy_root.string() << std::endl;
            std::filesystem::create_directories(bndy_root);
        }

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        // -- get save filename --
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << bndy_ix << ".ply"; // "%05d.ply" % bndy_ix
        std::filesystem::path ply_file = bndy_root / ss.str();
        aos_to_soa(sp_params, params); // from spix_params -> SuperpixelParams3d
        thrust::device_vector<uint32_t> poly;
        thrust::device_vector<uint32_t> spix_sizes_csum;

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // -- get polygon from points --
        std::tie(poly,spix_sizes_csum) = poly_from_points(data,params,border);

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
        thrust::host_vector<uint32_t> poly_cpu = poly;
        thrust::host_vector<uint32_t> spix_sizes_cpu = spix_sizes_csum;

        // -- unpack info --
        ScanNetScene scene;
        scene.write_spix_ply(ply_file,mu_app,mu_pos,var_pos,cov_pos,poly_cpu,spix_sizes_cpu,nspix);

        bx++;

    }

    // -- increment counter --
    bndy_ix++;

}