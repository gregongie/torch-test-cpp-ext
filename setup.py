from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='test_cpp',
      ext_modules=[cpp_extension.CppExtension('test_cpp', ['test.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
