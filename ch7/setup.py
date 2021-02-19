from distutils.core import setup

from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import glob
import shutil
from distutils.extension import Extension

__library_file__ = './lib/testpy*.so'
__version__ = '0.0.1'


class CopyLibFile(install):
    """"
    Directly copy library file to python's site-packages directory.
    """

    def run(self):
        install_dir = get_python_lib()
        lib_file = glob.glob(__library_file__)
        assert len(lib_file) == 1

        print('copying {} -> {}'.format(lib_file[0], install_dir))
        shutil.copy(lib_file[0], install_dir)


setup(name="PackageName",
      ext_modules=[
          Extension("testc", ["test2.cpp"],
                    libraries=["boost_python"])
      ])
