


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




void Logger::boundary_update(PointCloudData& in_data, SuperpixelParams3d& params, spix_params* sp_params) {

    // -- keep edges only --
    PointCloudData data = in_data.copy();
    data.labels = params.spix;
    data.border = params.border;
    // cudaMemset(data.border_ptr(), 0, data.V*sizeof(bool));
    // set_border(data.labels_ptr(), data.border_ptr(), data.csr_edges_ptr(),data.csr_eptr_ptr(),data.V);
    //filter_to_border_edges(data);

    // -- prepare superpixel information --
    aos_to_soa(sp_params, params); // from spix_params -> SuperpixelParams3d
    apply_spix_pooling(data,params);

    int bx = 0;
    // printf("log_roots.size(): %d\n",log_roots.size());
    for (const auto& log_root_s : log_roots) {

        // -- get save root --
        std::filesystem::path bndy_root = log_root_s / "bndy/";
        if (!std::filesystem::exists(bndy_root)) {
            std::cout << "Creaing output path " + bndy_root.string() << std::endl;
            std::filesystem::create_directories(bndy_root);
        }

        // -- get save filename --
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << bndy_ix << "_spix.ply"; // "%05d.ply" % bndy_ix
        std::filesystem::path ply_file = bndy_root / ss.str();

        // -- get batch slice --
        SuperpixelParams3dHost host_spix(params,bx);
 
        // -- write superpixel information --
        ScanNetScene scene;
        bool tmp = scene.write_spix_ply(ply_file,host_spix);//mu_app,mu_pos,var_pos,cov_pos,nspix);
        //bool tmp;

       // -- save boundary edges --
        std::stringstream ss_scene;
        ss_scene << std::setfill('0') << std::setw(5) << bndy_ix << "_edges.ply";  // "%05d.ply" % bndy_ix
        std::filesystem::path scene_fn = bndy_root / ss_scene.str();

        //-- write scene with only edges --
        PointCloudDataHost host_data(data,bx);
        scene.write_ply(scene_fn,host_data);

        bx++;

    }

    // -- increment counter --
    bndy_ix++;

}