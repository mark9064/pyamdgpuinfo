from libc.stdint cimport uint32_t, uint64_t

cdef extern from "amdgpu.h" nogil:
    struct amdgpu_device
    ctypedef amdgpu_device *amdgpu_device_handle
    struct amdgpu_gpu_info:
        uint32_t asic_id
        uint32_t chip_rev
        uint32_t chip_external_rev
        uint32_t family_id
        uint64_t ids_flags
        uint64_t max_engine_clk
        uint64_t max_memory_clk
        uint32_t num_shader_engines
        uint32_t num_shader_arrays_per_engine
        uint32_t avail_quad_shader_pipes
        uint32_t max_quad_shader_pipes
        uint32_t cache_entries_per_quad_pipe
        uint32_t num_hw_gfx_contexts
        uint32_t rb_pipes
        uint32_t enabled_rb_pipes_mask
        uint32_t gpu_counter_freq
        uint32_t backend_disable[4]
        uint32_t mc_arb_ramcfg
        uint32_t gb_addr_cfg
        uint32_t gb_tile_mode[32]
        uint32_t gb_macro_tile_mode[16]
        uint32_t pa_sc_raster_cfg[4]
        uint32_t pa_sc_raster_cfg1[4]
        uint32_t cu_active_number
        uint32_t cu_ao_mask
        uint32_t cu_bitmap[4][4]
        uint32_t vram_type
        uint32_t vram_bit_width
        uint32_t ce_ram_size
        uint32_t vce_harvest_config
        uint32_t pci_rev_id

    int amdgpu_device_initialize(int fd, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle)
    int amdgpu_read_mm_registers(amdgpu_device_handle dev, unsigned dword_offset, unsigned count, uint32_t instance, uint32_t flags, uint32_t *values)
    int amdgpu_query_info(amdgpu_device_handle dev, unsigned info_id, unsigned size, void *value)
    int amdgpu_query_gpu_info(amdgpu_device_handle dev, amdgpu_gpu_info *info)
    int amdgpu_query_sensor_info(amdgpu_device_handle dev, unsigned sensor_type, unsigned size, void *value)


cdef extern from "amdgpu_drm.h" nogil:
    int AMDGPU_INFO_VRAM_GTT "AMDGPU_INFO_VRAM_GTT"
    int AMDGPU_INFO_VRAM_USAGE "AMDGPU_INFO_VRAM_USAGE"
    int AMDGPU_INFO_GTT_USAGE "AMDGPU_INFO_GTT_USAGE"
    int AMDGPU_INFO_SENSOR_GFX_SCLK	"AMDGPU_INFO_SENSOR_GFX_SCLK"
    int AMDGPU_INFO_SENSOR_GFX_MCLK	"AMDGPU_INFO_SENSOR_GFX_MCLK"
    int AMDGPU_INFO_SENSOR_GPU_TEMP	"AMDGPU_INFO_SENSOR_GPU_TEMP"
    int AMDGPU_INFO_SENSOR_GPU_LOAD	"AMDGPU_INFO_SENSOR_GPU_LOAD"
    int AMDGPU_INFO_SENSOR_GPU_AVG_POWER "AMDGPU_INFO_SENSOR_GPU_AVG_POWER"
    struct drm_amdgpu_info_vram_gtt:
        uint64_t vram_size
        uint64_t vram_cpu_accessible_size
        uint64_t gtt_size
