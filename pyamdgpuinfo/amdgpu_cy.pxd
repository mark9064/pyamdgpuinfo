from libc.stdint cimport uint32_t, uint64_t

cdef extern from "amdgpu.h" nogil:
    struct amdgpu_device
    ctypedef amdgpu_device *amdgpu_device_handle
    # note: amdgpu_gpu_info defines more fields, we define only the ones needed
    # ordering does not matter as they are resolved by the compiler with the real header file
    struct amdgpu_gpu_info:
        uint64_t max_engine_clk
        uint64_t max_memory_clk

    int amdgpu_device_initialize(int fd, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle)
    int amdgpu_device_deinitialize(amdgpu_device_handle dev)
    int amdgpu_read_mm_registers(amdgpu_device_handle dev, unsigned dword_offset, unsigned count, uint32_t instance, uint32_t flags, uint32_t *values)
    int amdgpu_query_info(amdgpu_device_handle dev, unsigned info_id, unsigned size, void *value)
    int amdgpu_query_gpu_info(amdgpu_device_handle dev, amdgpu_gpu_info *info)
    int amdgpu_query_sensor_info(amdgpu_device_handle dev, unsigned sensor_type, unsigned size, void *value)
    const char *amdgpu_get_marketing_name(amdgpu_device_handle dev)


cdef extern from "amdgpu_drm.h" nogil:
    int AMDGPU_INFO_VRAM_USAGE "AMDGPU_INFO_VRAM_USAGE"
    int AMDGPU_INFO_GTT_USAGE "AMDGPU_INFO_GTT_USAGE"
    int AMDGPU_INFO_VRAM_GTT "AMDGPU_INFO_VRAM_GTT"
    int AMDGPU_INFO_SENSOR_GFX_SCLK	"AMDGPU_INFO_SENSOR_GFX_SCLK"
    int AMDGPU_INFO_SENSOR_GFX_MCLK	"AMDGPU_INFO_SENSOR_GFX_MCLK"
    int AMDGPU_INFO_SENSOR_GPU_TEMP	"AMDGPU_INFO_SENSOR_GPU_TEMP"
    int AMDGPU_INFO_SENSOR_GPU_LOAD	"AMDGPU_INFO_SENSOR_GPU_LOAD"
    int AMDGPU_INFO_SENSOR_GPU_AVG_POWER "AMDGPU_INFO_SENSOR_GPU_AVG_POWER"
    int AMDGPU_INFO_SENSOR_VDDNB "AMDGPU_INFO_SENSOR_VDDNB"
    int AMDGPU_INFO_SENSOR_VDDGFX "AMDGPU_INFO_SENSOR_VDDGFX"
    struct drm_amdgpu_info_vram_gtt:
        uint64_t vram_size
        uint64_t vram_cpu_accessible_size
        uint64_t gtt_size
