
cmake_minimum_required(VERSION 3.22)
project(Superpixels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Modern CUDA setup
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Set CUDA flags the modern way
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES "75" "86")  # Much cleaner than gencode
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Check if the custom OpenCV build exists
if(EXISTS "/home/gauenk/Documents/packages/opencv-py10.0/build/")
    # message(STATUS "Using custom OpenCV build")
    set(OpenCV_DIR "/home/gauenk/Documents/packages/opencv-py10.0/build/")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_DIR} include)
else()
    # message(STATUS "Using system OpenCV")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(SHARED_SOURCES
  bist/csrc/scannet_reader.cu
  bist/csrc/compact_spix_3d.cu
  bist/csrc/extract_edges.cu
  bist/csrc/main_utils.cu
  bist/csrc/file_io.cpp
  bist/csrc/logger.cu
  bist/csrc/bist.cu
  bist/csrc/init_utils.cu
  bist/csrc/init_seg.cu
  bist/csrc/init_sparams.cu
  bist/csrc/rgb2lab.cu
  bist/csrc/compact_spix.cu
  bist/csrc/compact_spix_cub.cu
  bist/csrc/seg_utils.cu
  bist/csrc/update_params.cu
  bist/csrc/update_seg.cu
  bist/csrc/split_merge.cu
  bist/csrc/split_merge_orig.cu
  bist/csrc/split_merge_prop.cu
  bist/csrc/sparams_io.cu
  bist/csrc/shift_and_fill.cu
  bist/csrc/shift_labels.cu
  bist/csrc/fill_missing.cu
  bist/csrc/sp_pooling.cu
  bist/csrc/split_disconnected.cu
  bist/csrc/relabel.cu
  bist/csrc/lots_of_merge.cu
)

# ADD THESE LINES - Explicitly mark .cu files as CUDA
set_source_files_properties(
    bist/csrc/main.cu
    bist/csrc/main3d.cu
    bist/csrc/main_utils.cu
    bist/csrc/scannet_reader.cu
    bist/csrc/compact_spix_3d.cu
    bist/csrc/extract_edges.cu
    bist/csrc/logger.cu
    bist/csrc/bass.cu
    bist/csrc/bass3d.cu
    bist/csrc/bist.cu
    bist/csrc/init_utils.cu
    bist/csrc/init_seg.cu
    bist/csrc/init_sparams.cu
    bist/csrc/rgb2lab.cu
    bist/csrc/compact_spix.cu
    bist/csrc/compact_spix_cub.cu
    bist/csrc/seg_utils.cu
    bist/csrc/update_params.cu
    bist/csrc/update_seg.cu
    bist/csrc/split_merge.cu
    bist/csrc/split_merge_orig.cu
    bist/csrc/split_merge_prop.cu
    bist/csrc/sparams_io.cu
    bist/csrc/shift_and_fill.cu
    bist/csrc/shift_labels.cu
    bist/csrc/fill_missing.cu
    bist/csrc/sp_pooling.cu
    bist/csrc/split_disconnected.cu
    bist/csrc/relabel.cu
    bist/csrc/lots_of_merge.cu
    PROPERTIES LANGUAGE CUDA
)

# 2D
add_executable(bist bist/csrc/main.cu ${SHARED_SOURCES} bist/csrc/bass.cu)
target_link_libraries(bist ${OpenCV_LIBS} cuda cublas cudadevrt) # CUDA linking
set_target_properties(bist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

# 3D
add_executable(bist3d bist/csrc/main3d.cu ${SHARED_SOURCES} bist/csrc/bass3d.cu)
target_link_libraries(bist3d ${OpenCV_LIBS} cuda cublas cudadevrt) # CUDA linking
set_target_properties(bist3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
