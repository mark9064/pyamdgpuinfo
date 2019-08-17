# pyamdgpuinfo

AMD GPU stats

## Install

`pip3 install pyamdgpuinfo`

Only linux is supported, using the amdgpu driver.

The library is written using cython, meaning that cython and and a C compiler are needed to build and install from source. Additionally, libdrm-dev is needed.

## Usage

Example:
```python
import pyamdgpuinfo

devices = pyamdgpuinfo.setup_devices()
device_stats = pyamdpuinfo.get_device_stats()
print(device_stats["/dev/dri/renderD129"]["temperature"])
```

All documentation is in the docstrings of each function/class.

Available functions are (see docstrings for more info):
* setup_devices - Sets up devices so they can be used.
* start_polling - Starts a polling thread to provide extra GPU utilisation info.
* stop_polling - Stops the polling thread.
* get_device_stats - Fetches GPU info.
* cleanup - Cleans up allocated memory (only recommended if de-initialising the module before the main program is ended).

## Mentions

Parts of this package were inspired by [radeontop](https://github.com/clbr/radeontop).


## License

GPLV3
