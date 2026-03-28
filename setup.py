"""Setup information"""

import setuptools
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

EXTENSIONS = [
    setuptools.Extension(
        name="pyamdgpuinfo._impl",
        sources=["pyamdgpuinfo/_impl.pyx"],
        include_dirs=["/usr/include/libdrm"],
        libraries=["drm_amdgpu"],
        extra_compile_args=["-DCYTHON_USE_TYPE_SPECS=1"],
    )
]

setuptools.setup(
    name="pyamdgpuinfo",
    version="2.1.7",
    author="mark9064",
    description="AMD GPU stats",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mark9064/pyamdgpuinfo",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    ext_modules=cythonize(EXTENSIONS),
    zip_safe=False,
    license="GPL-3.0-only",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
    ],
)
