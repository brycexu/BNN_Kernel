import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))
sources = []
headers = []

sources += ['src/binop_cpu_comp.c']
sources += ['src/binop_cpu_comp_kernel.c']
headers += ['include/binop_cpu_comp.h']
headers += ['include/binop_cpu_comp_kernel.h']

ffi = create_extension(
    'binop_cpu_comp',
    headers=headers,
    sources=sources,
    relative_to=__file__,
    include_dirs=[osp.join(abs_path, 'include')],
    extra_compile_args=["-std=c99", "-Ofast", "-fopenmp", "-mtune=native", "-march=x86-64"]
)

if __name__ == '__main__':
    ffi.build()
