# pyamdgpuinfo

AMD GPU information

## Install

`pip install pyamdgpuinfo`

Only Linux is supported, using the AMDGPU driver.

The library is written using Cython, meaning that Cython and and a C compiler are needed to build and install from source. Additionally, libdrm development headers are required. 

Precompiled wheels for Python 3.9-3.13 are the default method of install. This means that you don't need Cython or any other dependencies to install from PyPi.

## Usage

Example:
```python
>>> import pyamdgpuinfo
>>> pyamdgpuinfo.detect_gpus()
1 # we have 1 device present, so it'll be at index 0
>>> first_gpu = pyamdgpuinfo.get_gpu(0) # returns a GPUInfo object
>>> vram_usage = first_gpu.query_vram_usage()
>>> vram_usage
3954978816 # number of bytes in use
```

All documentation is in the docstrings of each function/class.

Available functions are (see docstrings for more info):
* detect_gpus - Returns the number of GPUs available
* get_gpu - Returns a GPUInfo object for the device index specified


GPUInfo methods (see docstring for class overview)
* start_utilisation_polling - Starts polling GPU functional units for utilisation statistics
* stop_utilisation_polling - Stops utilisation polling
* query_utilisation - Queries utilisation of different GPU parts
* query_max_clocks - Queries max GPU clocks
* query_sclk - Queries shader (core) clock
* query_mclk - Queries memory clock
* query_vram_usage - Queries VRAM usage
* query_gtt_usage - Queries GTT usage
* query_temperature - Queries temperature
* query_load - Queries GPU load
* query_power - Queries power consumption
* query_northbridge_voltage - Queries northbridge voltage
* query_graphics_voltage - Queries graphics voltage


VRAM and GTT sizes are available as an attribute of GPUInfo.

## Mentions

Parts of this package were inspired by [radeontop](https://github.com/clbr/radeontop).

## License

GPLV3
