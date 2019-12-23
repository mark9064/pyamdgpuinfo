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
# search path
cdef str DEVICE_PATH = "/dev/dri/"

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


cdef list find_devices():
    """Finds devices in DEVICE_PATH"""
    cdef:
        list devices = os.listdir(DEVICE_PATH)
        list detected_devices
    # attempt to use newer (non-root) renderD api
    detected_devices = [x for x in devices if "renderD" in x]
    # fallback (requires root on some systems)
    if not detected_devices:
        detected_devices = [x for x in devices if "card" in x]
    return sorted(detected_devices)


cpdef object setup_gpus():
    """Sets up amdgpu GPUs

    Params:
        No params.
    Raises:
        OSError: if an amdgpu GPU is found but cannot be initialised.
    Returns:
        collections.OrderedDict containing each GPU found as the key,
        and what features the GPU supports as the value, along with the VRAM and GTT size.
    Extra information:
        A GPU will only be present in the return value if it was initialised successfully.
        If there are no GPUs returned, attempting to use any of the other functions
        will result in an error (apart from cleanup() which will always succeed).
    """
    cdef:
        list devices
        object gpus = OrderedDict()
        xf86drm_cy.drmVersionPtr ver
        uint32_t rreg
        amdgpu_cy.amdgpu_device_handle* amdgpu_devices_init
        amdgpu_cy.amdgpu_device_handle* amdgpu_devices
        amdgpu_cy.amdgpu_device_handle amdgpu_dev
        amdgpu_cy.amdgpu_gpu_info gpu
        amdgpu_cy.drm_amdgpu_info_vram_gtt vram_gtt
        uint32_t major_ver, minor_ver
        int drm_fd
        unsigned long long out = 0
        int index
        int insert_index
        dict gpu_info
        str gpu_path
        str drm_name
    devices = find_devices()
    amdgpu_devices_init = (<amdgpu_cy.amdgpu_device_handle*>
                           calloc(len(devices), sizeof(amdgpu_cy.amdgpu_device_handle)))
    for index, gpu_path in enumerate(devices):
        gpu_info = {}
        try:
            drm_fd = os.open(os.path.join(DEVICE_PATH, gpu_path), os.O_RDWR)
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
            raise OSError("Can't initialize amdgpu driver for amdgpu GPU"
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

        gpus[gpu_path] = gpu_info
        amdgpu_devices_init[index] = amdgpu_dev

    if gpus:
        amdgpu_devices = (<amdgpu_cy.amdgpu_device_handle*>
                          calloc(len(gpus), sizeof(amdgpu_cy.amdgpu_device_handle)))
        insert_index = 0
        for index, gpu_path in enumerate(devices):
            if amdgpu_devices_init[index]:
                amdgpu_devices[insert_index] = amdgpu_devices_init[index]
                gpus[gpu_path]["id"] = insert_index
                insert_index += 1
        GLOBALS["gpus"] = <intptr_t>amdgpu_devices
    free(amdgpu_devices_init)
    GLOBALS["n_gpus"] = len(gpus)
    GLOBALS["py_gpus"] = gpus
    return gpus


cpdef object start_utilisation_polling(int ticks_per_second=100, int buffer_size_in_ticks=100):
    """Starts polling GPU registers for utilisation statistics

    Params:
        ticks_per_second (optional, default 100): int; specifies the number of device polls
        that will be performed each second.
        buffer_size_in_ticks (optional, default 100): int; specifies the number of ticks
        which will be stored.
    Raises:
        RuntimeError: if no GPUs are available.
        OSError: if the polling thread fails to start.
    Returns:
        Nothing returned.
    Extra information:
        The polling thread does not need to be started to get GPU statistics.
        The thread only provides stats on the utilisation of the different parts
        of the GPU (e.g shader interpolator, texture addresser).
        The time window used for these stats when calculating them is defined
        by the ticks_per_second and buffer_size_in_ticks args.
        When the utilisation polling thread is started, it takes some time to 'warm up'.
        While it is initialising, GPU utilisation cannot be queried and will result in
        an error. The length of this initialisation period is roughly equal to the buffer size
        in ticks divided by the ticks per second.

        Utilisation polling may not work correctly with all devices, in testing it
        has been unreliable with APUs.
    """
    cdef:
        pthread_cy.pthread_t tid
        pthread_cy.pthread_attr_t attr
        int index
        poll_args_t *args
        amdgpu_cy.amdgpu_device_handle *amdgpu_devices
    if "gpus" not in GLOBALS:
        raise RuntimeError("No GPUs available")
    pthread_cy.pthread_attr_init(&attr)
    pthread_cy.pthread_attr_setdetachstate(&attr, pthread_cy.PTHREAD_CREATE_DETACHED)
    amdgpu_devices = <amdgpu_cy.amdgpu_device_handle *><intptr_t>GLOBALS["gpus"]
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
        stop_utilisation_polling()
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


cpdef object stop_utilisation_polling():
    """Stops the utilisation polling thread

    Params:
        No params.
    Raises:
        RuntimeError: if the thread was not started.
    Returns:
        Nothing returned.
    Extra information:
        This also frees all resources allocated for the thread.
    """
    cdef:
        poll_args_t* args
        int index
    thread_args = GLOBALS.pop("thread_args", None)
    if thread_args is None:
        raise RuntimeError("Thread never started")
    args = <poll_args_t*><intptr_t>thread_args
    args.desired_state = 0
    # wait for it to exit
    while args.current_state != 0:
        usleep(1000)
    for index in range(GLOBALS["n_gpus"]):
        free(args.gpus[index].results)
    free(args.gpus)
    free(args)
    # no join required as it is detached


cpdef object cleanup():
    """Cleans up resources allocated for each GPU

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
    if "thread_args" in GLOBALS:
        stop_utilisation_polling()
    GLOBALS.pop("py_gpus", None)
    devices_pointer = GLOBALS.pop("gpus", None)
    if devices_pointer is not None:
        gpus = <amdgpu_cy.amdgpu_device_handle*><intptr_t>devices_pointer
        free(gpus)

cdef object check_query_valid(str gpu):
    if "gpus" not in GLOBALS:
        raise RuntimeError("GPUs not initialised, call setup_gpus() before querying a GPU")
    if gpu not in GLOBALS["py_gpus"]:
        raise RuntimeError("GPU requested does not exist / has not been initialised")

cdef amdgpu_cy.amdgpu_device_handle get_gpu_handle(str gpu):
    cdef:
        dict gpu_obj
        amdgpu_cy.amdgpu_device_handle gpu_handle
    gpu_obj = GLOBALS["py_gpus"][gpu]
    gpu_handle = (<amdgpu_cy.amdgpu_device_handle*><intptr_t>GLOBALS["gpus"])[gpu_obj["id"]]
    return gpu_handle

cdef object check_amdgpu_retcode(int retcode):
    if retcode != 0:
        raise RuntimeError("Unknown query failure (an amdgpu call failed)")

cpdef object query_max_clocks(str gpu):
    """Queries max GPU clocks
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        dict,
            - "max_sclk": int, hertz
            - "max_mclk": int, hertz
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        int multiplier = 10 ** 3
        amdgpu_cy.amdgpu_gpu_info gpu_info_struct
    check_query_valid(gpu)
    check_amdgpu_retcode(amdgpu_cy.amdgpu_query_gpu_info(get_gpu_handle(gpu), &gpu_info_struct))
    return {k: v * multiplier for k, v in dict(sclk_max=gpu_info_struct.max_engine_clk,
                                               mclk_max=gpu_info_struct.max_memory_clk).items()}
    
cpdef object query_vram_usage(str gpu):
    """Queries VRAM usage
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        int, usage in bytes.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        int multiplier = 1
    check_query_valid(gpu)
    check_amdgpu_retcode(query_info(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_VRAM_USAGE, &out))
    return out * multiplier

cpdef object query_gtt_usage(str gpu):
    """Queries GTT usage
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        int, usage in bytes.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        int multiplier = 1
    check_query_valid(gpu)
    check_amdgpu_retcode(query_info(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_GTT_USAGE, &out))
    return out * multiplier

cpdef object query_sclk(str gpu):
    """Queries shader (core) clock
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        int, shader clock in hertz.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        int multiplier = 10 ** 6
    check_query_valid(gpu)
    check_amdgpu_retcode(query_sensor(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_SCLK, &out))
    return out * multiplier

cpdef object query_mclk(str gpu):
    """Queries memory clock
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        int, memory clock in hertz.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        int multiplier = 10 ** 6
    check_query_valid(gpu)
    check_amdgpu_retcode(query_sensor(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_MCLK, &out))
    return out * multiplier

cpdef object query_temp(str gpu):
    """Queries temperature
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        float, temperature in °C.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        double multiplier = 1 / 10 ** 3
    check_query_valid(gpu)
    check_amdgpu_retcode(query_sensor(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_TEMP, &out))
    return out * multiplier

cpdef object query_load(str gpu):
    """Queries GPU load
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        float, value between 0 and 1.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        double multiplier = 1 / 10 ** 2
    check_query_valid(gpu)
    check_amdgpu_retcode(query_sensor(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_LOAD, &out))
    return out * multiplier

cpdef object query_power(str gpu):
    """Queries power consumption
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        int, consumption in watts.
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised.
            - An error when calling amdgpu methods.
    """
    cdef:
        unsigned long long out = 0
        int multiplier = 1
    check_query_valid(gpu)
    check_amdgpu_retcode(query_sensor(get_gpu_handle(gpu), amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_AVG_POWER, &out))
    return out * multiplier

cpdef object query_utilisation(str gpu):
    """Queries utilisation of different GPU parts
    
    Params:
        gpu: str; the name of the GPU to query as returned by setup_gpus().
    Raises:
        RuntimeError: details of the error will be in the error message (see extra infomation).
    Returns:
        dict containing utilisations from 0 - 1 (float):
            - "event_engine"
            - "texture_addresser"
            - "vertex_grouper_tesselator"
            - "shader_export"
            - "sequencer_instruction_cache"
            - "shader_interpolator"
            - "scan_converter"
            - "primitive_assembly"
            - "depth_block"
            - "colour_block"
            - "graphics_pipe"
    Extra information:
        RuntimeError can be raised for:
            - GPUs not being initialised
            - Utilisation polling thread not existing/alive if utilisation requested
            - Utilisation polling thread warming up
    """
    cdef:
        dict gpu_obj
        poll_args_t* thread_args
        dict utilisation
        int index
        uint32_t gbrm_value
        str bit
        unsigned int compare
    check_query_valid(gpu)
    gpu_obj = GLOBALS["py_gpus"][gpu]
    if "thread_args" not in GLOBALS:
        raise RuntimeError("Utilisation polling thread not running, call "
                           "start_utilisation_polling() (and wait for thread warmup) "
                           "before querying utilisation")
    thread_args = <poll_args_t*><intptr_t>GLOBALS["thread_args"]
    if thread_args.current_state == 0:
        raise RuntimeError("Utilisation polling thread is not alive")
    if thread_args.current_state == 2:
        raise RuntimeError("Utilisation polling thread warming up, "
                           "warmup takes {0:.2f}s with current options"
                           .format(thread_args.buffer_size / thread_args.ticks_per_second))
    utilisation = {k: 0 for k in COMPARE_BITS}
    for index in range(thread_args.buffer_size):
        gbrm_value = thread_args.gpus[gpu_obj["id"]].results[index]
        for bit, compare in COMPARE_BITS.items():
            if gbrm_value & compare:
                utilisation[bit] += 1
    return {k: v / thread_args.buffer_size for k, v in utilisation.items()}
