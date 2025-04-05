from ._impl import GPUInfo, get_gpu, detect_gpus

# update the module paths to avoid pyamdgpuinfo._impl.obj names
for value in locals().copy().values():
    if getattr(value, "__module__", "").startswith("pyamdgpuinfo."):
        value.__module__ = __name__
# make sure value doesn't hang around as a module attribute
del value  # pylint: disable=undefined-loop-variable