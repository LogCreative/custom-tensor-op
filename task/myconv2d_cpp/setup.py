from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='myconv2d_cpp',
      ext_modules=[cpp_extension.CppExtension('myconv2d_cpp', ['myconv2d.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})