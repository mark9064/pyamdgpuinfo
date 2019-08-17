"""GPU information using amdgpu"""
cimport pyamdgpuinfo.xf86drm_cy as xf86drm_cy
cimport pyamdgpuinfo.amdgpu_cy as amdgpu_cy
cimport pyamdgpuinfo.pthread_cy as pthread_cy

from libc.stdint cimport uint32_t, intptr_t
from libc.stdlib cimport malloc, calloc, free
from posix.time cimport CLOCK_MONOTONIC, timespec, clock_gettime
from posix.unistd cimport usleep, useconds_t

import os
from collections import OrderedDict

# const for gpu register queries
cdef int REGISTER_OFFSET = 8196

cdef struct gpu_device:
    uint32_t* results
    amdgpu_cy.amdgpu_device_handle* amdgpu_dev

cdef struct poll_args_t:
    # the state the thread should be in
    int desired_state
    # the thread's actual state
    int current_state # 0=dead, 1=alive, 2=warmup
    int n_gpus
    int ticks_per_second
    int buffer_size
    gpu_device* gpus


GLOBALS = {}
# comparison bits for the gpu register query
COMPARE_BITS = {
	"event_engine": 2 ** 10,
	"texture_addresser": 2 ** 14,
	"vertex_grouper_tesselator": 2 ** 16 | 2 ** 17,
	"shader_export": 2 ** 20,
	"sequencer_instruction_cache": 2 ** 21,
	"shader_interpolator": 2 ** 22,
	"scan_converter": 2 ** 24,
	"primitive_assembly": 2 ** 25,
	"depth_block": 2 ** 26,
	"colour_block": 2 ** 30,
	"graphics_pipe": 2 ** 31
}


cdef inline double get_time() nogil:
    """Returns monotonic time as a double"""
    cdef:
        timespec time_now
    if clock_gettime(CLOCK_MONOTONIC, &time_now) != 0:
        return -1
    return time_now.tv_sec + (time_now.tv_nsec / 10 ** 9)

cdef int get_registers(amdgpu_cy.amdgpu_device_handle amdgpu_dev, uint32_t *out) nogil:
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_read_mm_registers(amdgpu_dev, REGISTER_OFFSET, 1, 0xffffffff, 0, out)

cdef int query_info(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out):
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_info(amdgpu_dev, info, sizeof(unsigned long long), out)

cdef int query_sensor(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out):
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_sensor_info(amdgpu_dev, info, sizeof(unsigned long long), out)


cpdef object setup_devices():
    """Sets up amdgpu devices

    Params:
        No params.
    Raises:
        OSError: if an amdgpu device is found but cannot be initialised.
    Returns:
        collections.OrderedDict containing the paths of each device found as the key,
        and what features the device supports as the value.
    Extra information:
        A device will only be present in the return value if it was initialised successfully.
        If there are no devices returned, attempting to use any of the other functions
        will result in an error (apart from cleanup() which will always succeed).
    """
    cdef:
        list devices
        object out_devices = OrderedDict()
        xf86drm_cy.drmVersionPtr ver
        uint32_t rreg
        amdgpu_cy.amdgpu_device_handle* amdgpu_devices_init
        amdgpu_cy.amdgpu_device_handle* amdgpu_devices
        amdgpu_cy.amdgpu_device_handle amdgpu_dev
        amdgpu_cy.amdgpu_gpu_info gpu
        amdgpu_cy.drm_amdgpu_info_vram_gtt vram_gtt
        uint32_t major_ver, minor_ver
        int drm_fd
        unsigned long long out
        int index
        int insert_index
        dict gpu_info
        str gpu_path
        str drm_name
    devices = find_gpus()
    amdgpu_devices_init = (<amdgpu_cy.amdgpu_device_handle*>
                           calloc(len(devices), sizeof(amdgpu_cy.amdgpu_device_handle)))
    for index, gpu_path in enumerate(devices):
        gpu_info = {}
        try:
            drm_fd = os.open(gpu_path, os.O_RDWR)
        except Exception:
            continue
        ver = xf86drm_cy.drmGetVersion(drm_fd)
        drm_name = ver.name.decode()
        xf86drm_cy.drmFreeVersion(ver)
        if drm_name != "amdgpu":
            os.close(drm_fd)
            continue
        if amdgpu_cy.amdgpu_device_initialize(drm_fd, &major_ver, &minor_ver, &amdgpu_dev):
            os.close(drm_fd)
            raise OSError("Can't initialize amdgpu driver for amdgpu device"
                          " (this is usually a permission error)")
        os.close(drm_fd)
        # ioctl check
        if get_registers(amdgpu_dev, &rreg) != 0:
            continue

        if not amdgpu_cy.amdgpu_query_info(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_VRAM_GTT,
                                           sizeof(vram_gtt), &vram_gtt): # amdgpu vram gtt query succeeded
            gpu_info["vram_size"] = vram_gtt.vram_size
            gpu_info["gtt_size"] = vram_gtt.gtt_size
        gpu_info["vram_available"] = not query_info(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_VRAM_USAGE, &out)
        gpu_info["gtt_available"] = not query_info(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_GTT_USAGE, &out)
        gpu_info["clocks_minmax_available"] = not amdgpu_cy.amdgpu_query_gpu_info(amdgpu_dev, &gpu)
        gpu_info["sclk_available"] = not query_sensor(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_SCLK, &out)
        gpu_info["mclk_available"] = not query_sensor(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_MCLK, &out)
        gpu_info["temp_available"] = not query_sensor(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_TEMP, &out)
        gpu_info["load_available"] = not query_sensor(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_LOAD, &out)
        gpu_info["power_available"] = not query_sensor(amdgpu_dev, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_AVG_POWER, &out)

        out_devices[gpu_path] = gpu_info
        amdgpu_devices_init[index] = amdgpu_dev

    if out_devices:
        amdgpu_devices = (<amdgpu_cy.amdgpu_device_handle*>
                          calloc(len(out_devices), sizeof(amdgpu_cy.amdgpu_device_handle)))
        insert_index = 0
        for index, gpu_path in enumerate(devices):
            if amdgpu_devices_init[index]:
                amdgpu_devices[insert_index] = amdgpu_devices_init[index]
                out_devices[gpu_path]["id"] = insert_index
                insert_index += 1
        GLOBALS["devices"] = <intptr_t>amdgpu_devices
    free(amdgpu_devices_init)
    GLOBALS["n_gpus"] = len(out_devices)
    GLOBALS["py_devices"] = out_devices
    return out_devices


cpdef object start_polling(int ticks_per_second=100, int buffer_size_in_ticks=100):
    """Starts polling device registers for utilisation statistics

    Params:
        ticks_per_second (optional, default 100): int; specifies the number of device polls
        that will be performed each second.
        buffer_size_in_ticks (optional, default 100): int; specifies the number of ticks
        which will be stored.
    Raises:
        RuntimeError: if no devices are available.
        OSError: if the polling thread fails to start.
    Returns:
        Nothing returned.
    Extra information:
        The polling thread does not need to be started to get device statistics.
        The thread only provides stats on the utilisation of the different parts
        of the device (e.g shader interpolator, texture addresser).
        The time window used for these stats when calculating them is defined
        by the ticks_per_second and buffer_size_in_ticks args.
    """
    cdef:
        pthread_cy.pthread_t tid
        pthread_cy.pthread_attr_t attr
        int index
        poll_args_t *args
        amdgpu_cy.amdgpu_device_handle *amdgpu_devices
    if "devices" not in GLOBALS:
        raise RuntimeError("No devices available")
    pthread_cy.pthread_attr_init(&attr)
    pthread_cy.pthread_attr_setdetachstate(&attr, pthread_cy.PTHREAD_CREATE_DETACHED)
    amdgpu_devices = <amdgpu_cy.amdgpu_device_handle *><intptr_t>GLOBALS["devices"]
    args = <poll_args_t*>malloc(sizeof(poll_args_t))
    args.n_gpus = GLOBALS["n_gpus"]
    args.gpus = <gpu_device*>calloc(GLOBALS["n_gpus"], sizeof(gpu_device))
    for index in range(GLOBALS["n_gpus"]):
        args.gpus[index].results = <uint32_t*>calloc(buffer_size_in_ticks, sizeof(uint32_t))
        args.gpus[index].amdgpu_dev = &amdgpu_devices[index]
    args.desired_state = 1
    args.current_state = 0
    args.ticks_per_second = ticks_per_second
    args.buffer_size = buffer_size_in_ticks
    if pthread_cy.pthread_create(&tid, &attr, poll_registers, args) != 0:
        # cleans up
        stop_polling()
        raise OSError("Poll thread failed to start")
    GLOBALS["thread_args"] = <intptr_t>args


cdef void* poll_registers(void *arg) nogil:
    cdef:
        int index = 0
        poll_args_t *args
        int item
        uint32_t out
        double start_time
        double exec_time
    args = <poll_args_t*>arg
    args.current_state = 2
    while args.desired_state:
        start_time = get_time()
        if start_time == -1:
            args.current_state = 0
            return NULL
        if index >= args.buffer_size:
            args.current_state = 1
            index = 0
        for item in range(args.n_gpus):
            out = 0
            if get_registers(args.gpus[item].amdgpu_dev[0], &out) != 0:
                args.current_state = 0
                return NULL
            args.gpus[item].results[index] = out
        index += 1
        exec_time = get_time()
        if exec_time == -1:
            args.current_state = 0
            return NULL
        exec_time -= start_time
        # multiply by 10^6 for seconds to microseconds
        usleep(<useconds_t>(((1 / args.ticks_per_second) - exec_time) * 10 ** 6))
    args.current_state = 0
    return NULL


cpdef stop_polling():
    """Stops the polling thread

    Params:
        No params.
    Raises:
        RuntimeError: if the thread was not started
    Returns:
        Nothing returned.
    Extra information:
        This also frees all resources allocated for the thread."""
    cdef:
        poll_args_t* args
        int index
    if "thread_args" not in GLOBALS:
        raise RuntimeError("Thread never started")
    args = <poll_args_t*><intptr_t>GLOBALS.pop("thread_args")
    args.desired_state = 0
    # wait for it to exit
    while args.current_state != 0:
        usleep(1000)
    for index in range(GLOBALS["n_gpus"]):
        free(args.gpus[index].results)
    free(args.gpus)
    free(args)
    # no join required as it is detached


cpdef object get_device_stats(ignore=None):
    """Fetches stats for a device

    Params:
        ignore (optional, default None): list; specified devices to ignore by path.
    Raises:
        RuntimeError: if no devices are available.
    Returns:
        collections.OrderedDict containg path as key and device infomation as the value.
    Extra information:
        The data returned depends on what has been detected as available. This detection
        is done in setup_devices(). If the start_polling() has been called successfully,
        infomation on utilisation of different device parts will be available.
        The following values can be present:
            "vram_usage" - video memory usage (B)
            "vram_size" - total video memory (B)
            "gtt_usage" - graphics translation table (GTT) usage (B)
            "gtt_size" - total GTT (B)
            "sclk_max" - max shader (core) clock (Hz)
            "sclk_current" - current shader clock (Hz)
            "mclk_max" - max memory clock (Hz)
            "mclk_current" - current memory clock (Hz)
            "temperature" - temperature (°C)
            "load" - overall gpu load (0 - 1)
            "average_power" - power consumption (W)
            --if start_polling--
            "utilisation":
                "event_engine" (0 - 1)
                "texture_addresser" (0 - 1)
                "vertex_grouper_tesselator" (0 - 1)
                "shader_export" (0 - 1)
                "sequencer_instruction_cache" (0 - 1)
                "shader_interpolator" (0 - 1)
                "scan_converter" (0 - 1)
                "primitive_assembly" (0 - 1)
                "depth_block" (0 - 1)
                "colour_block" (0 - 1)
                "graphics_pipe" (0 - 1)
    """
    cdef:
        poll_args_t* thread_args
        int gpu_id
        int index
        object out_results = OrderedDict()
        unsigned long long out
        uint32_t gbrm_value
        amdgpu_cy.amdgpu_gpu_info gpu
        dict utilisation
        dict gpu_info
        str gpu_path
        str bit
        amdgpu_cy.amdgpu_device_handle* gpus
        unsigned int compare
        bint thread_exists = "thread_args" in GLOBALS
    if "devices" not in GLOBALS:
        raise RuntimeError("No devices available")
    gpus = <amdgpu_cy.amdgpu_device_handle*><intptr_t>GLOBALS["devices"]
    if thread_exists:
        thread_args = <poll_args_t*><intptr_t>GLOBALS["thread_args"]
    if ignore is None:
        ignore = []
    for gpu_path, gpu_info in GLOBALS["py_devices"].items():
        if gpu_path in ignore:
            continue
        out_results[gpu_path] = {}
        gpu_id = gpu_info["id"]
        if gpu_info["vram_available"]:
            query_info(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_VRAM_USAGE, &out)
            out_results[gpu_path]["vram_usage"] = out
        if "vram_size" in gpu_info:
            out_results[gpu_path]["vram_size"] = gpu_info["vram_size"]
        if gpu_info["gtt_available"]:
            query_info(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_GTT_USAGE, &out)
            out_results[gpu_path]["gtt_usage"] = out
        if "gtt_size" in gpu_info:
            out_results[gpu_path]["gtt_size"] = gpu_info["gtt_size"]
        if gpu_info["clocks_minmax_available"]:
            amdgpu_cy.amdgpu_query_gpu_info(gpus[gpu_id], &gpu)
            out_results[gpu_path]["sclk_max"] = gpu.max_engine_clk * 10 ** 3
            out_results[gpu_path]["mclk_max"] = gpu.max_memory_clk * 10 ** 3
        if gpu_info["sclk_available"]:
            query_sensor(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_SCLK, &out)
            out_results[gpu_path]["sclk_current"] = out * 10 ** 6
        if gpu_info["mclk_available"]:
            query_sensor(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_MCLK, &out)
            out_results[gpu_path]["mclk_current"] = out * 10 ** 6
        if gpu_info["temp_available"]:
            query_sensor(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_TEMP, &out)
            out_results[gpu_path]["temperature"] = out / 1000
        if gpu_info["load_available"]:
            query_sensor(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_LOAD, &out)
            out_results[gpu_path]["load"] = out / 100
        if gpu_info["power_available"]:
            query_sensor(gpus[gpu_id], amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_AVG_POWER, &out)
            out_results[gpu_path]["avg_power"] = out

        if thread_exists:
            if thread_args.current_state == 1:
                utilisation = {k: 0 for k in COMPARE_BITS}
                for index in range(thread_args.buffer_size):
                    gbrm_value = thread_args.gpus[gpu_id].results[index]
                    for bit, compare in COMPARE_BITS.items():
                        if gbrm_value & compare:
                            utilisation[bit] += 1
                out_results[gpu_path]["utilisation"] = (
                    {k: v / thread_args.buffer_size for k, v in utilisation.items()}
                )
    return out_results


cpdef cleanup():
    """Cleans up resources allocated for each device

    Params:
        No params.
    Raises:
        No specific exceptions.
    Returns:
        Nothing returned.
    Extra infomation:
        This should be ran when you are finished with using the library,
        but there's no point running it when your program finishes as the
        resources will be automatically freed when the process ends.
    """
    devices_pointer = GLOBALS.pop("devices", None)
    if devices_pointer is not None:
        gpus = <amdgpu_cy.amdgpu_device_handle*><intptr_t>devices_pointer
        free(gpus)


cdef list find_gpus():
    """Finds gpus"""
    cdef:
        str gpu_path = "/dev/dri/"
        list gpus = os.listdir(gpu_path)
        list detected_gpus
        int index
        str card
    # attempt to use newer (non-root) renderD api
    detected_gpus = [x for x in gpus if "renderD" in x]
    # fallback (requires root on some systems)
    if not detected_gpus:
        detected_gpus = [x for x in gpus if "card" in x]
    for index, card in enumerate(detected_gpus):
        detected_gpus[index] = os.path.join(gpu_path, card)
    return sorted(detected_gpus)
