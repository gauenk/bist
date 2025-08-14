cmake_minimum_required(VERSION 2.8)
project( Superpixels )

# Set specific GCC version for CUDA compatibility
set(CMAKE_C_COMPILER gcc-8)
set(CMAKE_CXX_COMPILER g++-8)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Enable ccache if available
# find_program(CCACHE_PROGRAM ccache)
# if(CCACHE_PROGRAM)
#     set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
#     set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
#     message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
# endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CUDA_SEPARABLE_COMPILATION ON)

# Build configuration
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA QUIET REQUIRED )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-O3" "--extended-lambda" "-gencode=arch=compute_75,code=compute_75" "-gencode=arch=compute_86,code=sm_86" "-gencode=arch=compute_86,code=compute_86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++14" "-O3" "--extended-lambda" "-gencode=arch=compute_86,code=sm_86" "-gencode=arch=compute_86,code=compute_86")

# find_package(OpenCV REQUIRED)
# SET(OpenCV_DIR "/home/gauenk/Documents/packages/opencv-py10.0/build/")
# INCLUDE_DIRECTORIES(${OpenCV_DIR} include)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

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

cuda_add_executable(bist bist/csrc/main.cu ${SHARED_SOURCES})
#add_executable(bist bist/csrc/main.cu ${SHARED_SOURCES})
target_link_libraries(bist ${OpenCV_LIBS} cuda cublas cudadevrt)
set_target_properties(bist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

