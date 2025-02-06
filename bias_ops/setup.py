from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# 移除 -Wstrict-prototypes 标志，因为它对 C/C++ 无效
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

# 获取当前文件的绝对路径
this_dir = os.path.dirname(os.path.abspath(__file__))


setup(
    name='biasops',
    packages=["biasops"],
    package_dir={"biasops": "functions"},
    ext_modules=[
        CUDAExtension(
            name='biasops._C',
            sources=[
                os.path.join(this_dir, 'src', 'bias_max.cpp'),
                os.path.join(this_dir, 'src', 'bias_max_kernel.cu'),
                os.path.join(this_dir, 'src', 'bias_query.cpp'),
                os.path.join(this_dir, 'src', 'bias_query_kernel.cu'),
                os.path.join(this_dir, 'src', 'biasops_api.cpp'),
            ],

            extra_compile_args={
                "cxx": ["-g", f"-I{os.path.join(this_dir, 'src')}"],
                "nvcc": ["-O2", f"-I{os.path.join(this_dir, 'src')}"],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
