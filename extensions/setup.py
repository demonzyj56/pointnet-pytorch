"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='dist_map_cuda',
    ext_modules=[
        CUDAExtension('dist_map_cuda', [
            'dist_map_cuda.cpp',
            'dist_map_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='bpq_cuda',
    ext_modules=[
        CUDAExtension('bpq_cuda', [
            'ball_point_query_cuda.cpp',
            'ball_point_query_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='fps_cuda',
    ext_modules=[
        CUDAExtension('fps_cuda', [
            'farthest_point_sample_cuda.cpp',
            'farthest_point_sample_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
