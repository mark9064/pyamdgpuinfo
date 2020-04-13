# pyamdgpuinfo

AMD GPU information

## Install

`pip3 install pyamdgpuinfo`

Only linux is supported, using the amdgpu driver.

Precompiled wheels for python 3.5, 3.6, 3.7 and 3.8 are the default method of install. This means that you don't need cython or any other dependencies to install it normally.

The library is written using cython, meaning that cython and and a C compiler are needed to build and install from source. Additionally, libdrm-dev is needed. 

## Usage

Example:
```python
>>> import pyamdgpuinfo
>>> n_devices = pyamdgpuinfo.detect_gpus()
1 # we have 1 device present, so it'll be at index 0
>>> first_gpu = pyamdgpuinfo.get_gpu(0) # returns a GPUInfo object
>>> vram_usage = first_gpu.query_vram_usage()
>>> print(vram_usage)
3954978816 # number of bytes in use
```

All documentation is in the docstrings of each function/class.

Available functions are (see docstrings for more info):
* detect_gpus - Returns the number of GPUs available
* get_gpu - Returns a GPUInfo object for the device index specified


GPUInfo methods (see docstring for class overview)
* start_utilisation_polling - Starts polling GPU registers for utilisation statistics
* stop_utilisation_polling - Stops the utilisation polling thread
* query_utilisation - Queries utilisation of different GPU parts
* query_max_clocks - Queries max GPU clocks
* query_sclk - Queries shader (core) clock
* query_mclk - Queries memory clock
* query_vram_usage - Queries VRAM usage
* query_gtt_usage - Queries GTT usage
* query_temperature - Queries temperature
* query_load - Queries GPU load
* query_power - Queries power consumption
* query_northbridge_voltage - Queries northbrige voltage
* query_graphics_voltage - Queries graphics voltage


VRAM and GTT sizes are available as an attribute of GPUInfo.

## Mentions

Parts of this package were inspired by [radeontop](https://github.com/clbr/radeontop).

## License

GPLV3
