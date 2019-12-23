"""Setup information"""
import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

EXTENSIONS = [Extension(name="pyamdgpuinfo.pyamdgpuinfo", sources=["pyamdgpuinfo/pyamdgpuinfo.pyx"],
                        include_dirs=["/usr/include/libdrm"], libraries=["drm_amdgpu"])]

setuptools.setup(
    name="pyamdgpuinfo",
    version="1.0.3",
    author="mark9064",
    description="AMD GPU stats",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mark9064/pyamdgpuinfo",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(EXTENSIONS, language_level=3),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
        "Natural Language :: English"
    ],
)
