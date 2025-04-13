
# -- local --
import os
# os.environ['TORCH_USE_CUDA_DSA'] = '1' # debug
# os.environ['PYTORCH_NVCC'] = "ccache nvcc"
# os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '1' # "1" # for faster
# os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '1' # "1" # for fasterb
os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '0' # "1" # for fasterb

# Make sure the bin directory exists
if not os.path.exists('bin'):
    os.makedirs('bin')
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bist",
    py_modules=["bist"],
    install_requires=[],
    package_dir={"": "."},
    packages=find_packages("."),
    package_data={'bin': ['*.so']},
    include_package_data=True,
    ext_modules=[

        # -- keep me --
        CUDAExtension('bin.bist_cuda', [
            # -- shared utils --
            "bist/csrc/pyapi.cu",
            "bist/csrc/init_utils.cu",
            "bist/csrc/init_seg.cu",
            "bist/csrc/init_sparams.cu",
            "bist/csrc/rgb2lab.cu",
            "bist/csrc/compact_spix.cu",
            "bist/csrc/seg_utils.cu",
            "bist/csrc/update_params.cu",
            "bist/csrc/update_seg.cu",
            "bist/csrc/split_merge.cu",
            "bist/csrc/split_merge_orig.cu",
            "bist/csrc/split_merge_prop.cu",
            "bist/csrc/sparams_io.cu",
            "bist/csrc/shift_and_fill.cu",
            "bist/csrc/shift_labels.cu",
            "bist/csrc/fill_missing.cu",
            "bist/csrc/sp_pooling.cu",
            "bist/csrc/split_disconnected.cu",
            "bist/csrc/relabel.cu",
            "bist/csrc/logger.cu",
            "bist/csrc/bass.cu",
            "bist/csrc/bist.cu",
            # -- pybind --
            "bist/csrc/pybind.cpp",
        ],
        libraries=['cuda', 'cublas', 'cudadevrt'],
        extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-w','--extended-lambda']},
        library_dirs=['bin'],),
    ],
    cmdclass={'build_ext': BuildExtension},
)
