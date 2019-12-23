# pyamdgpuinfo

AMD GPU stats

## Install

`pip3 install pyamdgpuinfo`

Only linux is supported, using the amdgpu driver.

Precompiled wheels for python 3.5, 3.6, 3.7 and 3.8 are the default method of install. This means that you don't need cython or any other dependencies to install it normally.

The library is written using cython, meaning that cython and and a C compiler are needed to build and install from source. Additionally, libdrm-dev is needed. 

## Usage

Example:
```python
import pyamdgpuinfo

gpus = pyamdgpuinfo.setup_devices()
# query first device
first_gpu = list(gpus.keys())[0]
vram_usage = pyamdpuinfo.query_vram_usage(first_gpu)
print(vram_usage)
```

All documentation is in the docstrings of each function/class.

Available functions are (see docstrings for more info):
* setup_devices - Sets up devices so they can be used.
* start_utilisation_polling - Starts polling GPU registers for utilisation statistics.
* stop_utilisation_polling - Stops the utilisation polling thread.
* cleanup - Cleans up allocated memory (only recommended if de-initialising the module before the main program is ended).

Query functions (again see docstrings):
* query_max_clocks - Queries max GPU clocks
* query_sclk - Queries shader (core) clock
* query_mclk - Queries memory clock
* query_vram_usage - Queries VRAM usage
* query_gtt_usage - Queries GTT usage
* query_temp - Queries temperature
* query_load - Queries GPU load
* query_power - Queries power consumption
* query_utilisation - Queries utilisation of different GPU parts (requires utilisation polling to be running)

VRAM and GTT sizes are returned by setup_devices (if they are available).

## Mentions

Parts of this package were inspired by [radeontop](https://github.com/clbr/radeontop).

## License

GPLV3
