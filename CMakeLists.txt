
cmake_minimum_required(VERSION 3.22)
project(Superpixels)

# Force use of CUDA 12.8
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

# Set specific GCC version for CUDA compatibility
# set(CMAKE_C_COMPILER gcc-8)
# set(CMAKE_CXX_COMPILER g++-8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)


# Remove these old CUDA lines:
# find_package(CUDA QUIET REQUIRED)
# find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS ...)
# set(CUDA_SEPARABLE_COMPILATION ON)

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
# set(OpenCV_DIR "/home/gauenk/Documents/packages/opencv-py10.0/build/")
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_DIR} include)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

set(SHARED_SOURCES
  bist/csrc/main_utils.cu
  bist/csrc/file_io.cpp
  bist/csrc/logger.cu
  bist/csrc/bass.cu
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
    bist/csrc/main_utils.cu
    bist/csrc/logger.cu
    bist/csrc/bass.cu
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

# Modern executable creation
add_executable(bist bist/csrc/main.cu ${SHARED_SOURCES})

# Modern CUDA linking
target_link_libraries(bist ${OpenCV_LIBS} 
    cuda cublas cudadevrt)

set_target_properties(bist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
