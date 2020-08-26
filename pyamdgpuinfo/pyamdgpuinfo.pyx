"""GPU information using amdgpu"""
cimport pyamdgpuinfo.xf86drm_cy as xf86drm_cy
cimport pyamdgpuinfo.amdgpu_cy as amdgpu_cy
cimport pyamdgpuinfo.pthread_cy as pthread_cy

from libc.stdint cimport uint32_t
from libc.stdlib cimport malloc, calloc, free
from posix.time cimport CLOCK_MONOTONIC, timespec, clock_gettime, nanosleep

import os
import time


# gpu registers location
cdef int GRBM_OFFSET = 8196
# search path
cdef str DEVICE_PATH = "/dev/dri/by-path/"
cdef str DEVICE_FALLBACK_PATH = "/dev/dri/"

COMPARE_BITS = {
    "texture_addresser": 2 ** 14,
    "shader_export": 2 ** 20,
    "shader_interpolator": 2 ** 22,
    "scan_converter": 2 ** 24,
    "primitive_assembly": 2 ** 25,
    "depth_block": 2 ** 26,
    "colour_block": 2 ** 30,
    "graphics_pipe": 2 ** 31
}


cdef int get_registers(amdgpu_cy.amdgpu_device_handle amdgpu_dev, uint32_t *out) nogil:
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_read_mm_registers(amdgpu_dev, GRBM_OFFSET, 1, 0xffffffff, 0, out)

cdef int query_info(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out):
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_info(amdgpu_dev, info, sizeof(unsigned long long), out)

cdef int query_sensor(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out):
    """Returns whether amdgpu query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_sensor_info(amdgpu_dev, info, sizeof(unsigned long long), out)


cdef timespec timespec_subtract(timespec left, timespec right) nogil:
    """Performs left - right"""
    cdef:
        timespec temp
    if left.tv_nsec - right.tv_nsec < 0:
        temp.tv_sec = left.tv_sec - right.tv_sec - 1
        temp.tv_nsec = 1000000000 + (left.tv_nsec - right.tv_nsec)
    else:
        temp.tv_sec = left.tv_sec - right.tv_sec
        temp.tv_nsec = left.tv_nsec - right.tv_nsec
    return temp

cdef timespec timespec_clamp(timespec time_to_clamp) nogil:
    """Clamps a timespec to 0 if seconds is negative"""
    if time_to_clamp.tv_sec < 0:
        time_to_clamp.tv_sec = 0
        time_to_clamp.tv_nsec = 0
    return time_to_clamp
    
cdef timespec timespec_sum(timespec left, timespec right) nogil:
    """Performs left + right"""
    cdef:
        timespec temp
    temp.tv_nsec = left.tv_nsec + right.tv_nsec
    temp.tv_sec = left.tv_sec + right.tv_sec
    if temp.tv_nsec >= 1000000000:
        temp.tv_sec += 1
        temp.tv_nsec -= 1000000000
    return temp


cdef struct poll_args_t:
    # the state the thread should be in
    int desired_state
    # the thread's actual state
    int current_state # 0=dead, 1=alive, 2=warmup
    timespec measurement_interval
    int buffer_size
    uint32_t* results
    amdgpu_cy.amdgpu_device_handle* device_handle


cdef void* poll_registers(void *arg) nogil:
    cdef:
        int index = 0
        poll_args_t *args
        uint32_t register_data
        timespec iteration_start_time
        timespec iteration_end_time
        timespec sleep_start_time
        timespec sleep_duration
    args = <poll_args_t*>arg
    args.current_state = 2
    # timer init
    sleep_duration = args.measurement_interval
    if clock_gettime(CLOCK_MONOTONIC, &sleep_start_time) != 0:
        args.current_state = 0
        return NULL
    nanosleep(&sleep_duration, NULL)
    while args.desired_state:
        clock_gettime(CLOCK_MONOTONIC, &iteration_start_time)
        if get_registers(args.device_handle[0], &register_data) != 0:
            args.current_state = 0
            return NULL
        args.results[index] = register_data
        index += 1
        if index >= args.buffer_size:
            args.current_state = 1
            index = 0
        clock_gettime(CLOCK_MONOTONIC, &iteration_end_time)
        sleep_duration = timespec_clamp(
            timespec_subtract(
                args.measurement_interval,
                timespec_sum(
                    # iteration time
                    timespec_subtract(
                        iteration_end_time,
                        iteration_start_time
                    ),
                    # oversleep
                    timespec_subtract(
                        # real sleep time
                        timespec_subtract(
                            iteration_start_time,
                            sleep_start_time
                        ),
                        # target sleep time
                        sleep_duration
                    )
                )
            )
        )
        clock_gettime(CLOCK_MONOTONIC, &sleep_start_time)
        nanosleep(&sleep_duration, NULL)
    args.current_state = 0
    return NULL


cdef list find_devices():
    """Finds devices in DEVICE_PATH"""
    cdef:
        list devices
        list detected_devices
        str path = DEVICE_PATH
    if not os.path.exists(path):
        path = DEVICE_FALLBACK_PATH
    devices = os.listdir(path)
    # attempt to use newer (non-root) renderD api
    detected_devices = [x for x in devices if "render" in x]
    # fallback (requires root on some systems)
    if not detected_devices:
        detected_devices = [x for x in devices if "card" in x]
    detected_devices = [os.path.join(path, x) for x in detected_devices]
    return sorted(detected_devices)


cpdef int detect_gpus():
    """Returns the number of GPUs available
    
    Params:
        No params.
    Raises:
        RuntimeError: if an error occurs when calling amdgpu methods.
    Returns:
        int; number of GPUs available.
    """
    return get_gpu(-1)


cpdef object get_gpu(int gpu_id):
    """Returns GPUInfo object for gpu_id

    Params:
        gpu_id: int; the sequential id of the GPU to get
    Raises:
        RuntimeError: if an error occurs when calling amdgpu methods.
    Returns:
        GPUInfo; object providing interface to GPU info calls
    Extra information:
        Find the number of GPUs available by calling detect_gpus() first.
        e.g you could do
            n = pyamdgpuinfo.detect_gpus()
            gpus = [pyamdgpuinfo.get_gpu(x) for x in range(n)]
    """
    cdef:
        list devices
        xf86drm_cy.drmVersionPtr ver
        uint32_t register_data
        amdgpu_cy.amdgpu_device_handle device_handle
        uint32_t major_ver, minor_ver
        int drm_fd
        int index
        int amdgpu_index = 0
        dict gpu_info
        str gpu_path
        str drm_name
    devices = find_devices()
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
        if amdgpu_cy.amdgpu_device_initialize(drm_fd, &major_ver, &minor_ver, &device_handle):
            os.close(drm_fd)
            raise OSError("Can't initialize amdgpu driver for amdgpu GPU"
                          " (this is usually a permission error)")
        os.close(drm_fd)
        # ioctl check
        if get_registers(device_handle, &register_data) != 0:
            continue
        amdgpu_cy.amdgpu_device_deinitialize(device_handle)
        if amdgpu_index == gpu_id:
            return GPUInfo(gpu_path, gpu_id)
        amdgpu_index += 1
    if gpu_id == -1:
        return amdgpu_index
    raise RuntimeError("GPU id {0} not found".format(gpu_id))


cdef class GPUInfo:
    """Interface for amdgpu device info queries

    Params:
        device_path: str; full path to device in /dev/dri/ or /dev/dri/by-path/.
        gpu_id: int; sequential GPU id (from get_gpu).
    Raises:
        No specific exceptions.
    Attrs:
        gpu_id: int; sequential GPU id.
        path: str; full path to GPU node.
        pci_slot: str, None; PCI slot of GPU (None if unavailable).
        name: str, None; marketing name of GPU (None if unavailable).
        memory_info: dict (either all values will be present or all will be None)
            - "vram_size": int, None; GPU VRAM size.
            - "gtt_size": int, None; GPU GTT size.
            - "vram_cpu_accessible_size": int, None; size of VRAM accessible by CPU.
    Methods:
        start_utilisation_polling: starts the utilisation polling thread.
        stop_utilisation_polling: stops the utilisation polling thread.
        query_utilisation: queries utilisation of different GPU units using polling thread.
        query_max_clocks: queries max GPU clocks.
        query_vram_usage: queries VRAM usage.
        query_gtt_usage: queries GTT usage.
        query_sclk: queries shader (core) clock.
        query_mclk: queries memory clock.
        query_temperature: queries temperature.
        query_load: queries overall GPU load.
        query_power: queries power consumption.
        query_northbridge_voltage: queries northbrige voltage.
        query_graphics_voltage: queries graphics voltage.
    Extra information:
        The pci_slot attr can be used to identify cards:
            - This is tied to a physical PCI slot 
            - It will only change if the card is moved to a different slot
              or if the motherboard is changed
        This class can be instantiated directly, but a typical application would be:
            - Find number of GPUs with detect_gpus()
            - Use get_gpu(index) which returns a GPUInfo object
    """
    cdef:
        readonly int gpu_id
        readonly str path
        readonly str pci_slot
        readonly str name
        readonly dict memory_info
        # private
        bint utilisation_polling
        amdgpu_cy.amdgpu_device_handle device_handle
        poll_args_t* thread_args

    def __cinit__(self):
        self.utilisation_polling = False

    def __dealloc__(self):
        if self.utilisation_polling:
            self.thread_args.desired_state = 0
            # wait for it to exit
            while self.thread_args.current_state != 0:
                time.sleep(0.001)
            free(self.thread_args.results)
            free(self.thread_args)
            self.utilisation_polling = False
        amdgpu_cy.amdgpu_device_deinitialize(self.device_handle)

    def __init__(self, device_path, gpu_id):
        cdef:
            amdgpu_cy.drm_amdgpu_info_vram_gtt vram_gtt
            char* device_name
            uint32_t major_ver, minor_ver
        self.gpu_id = gpu_id
        self.path = device_path
        if "by-path" in device_path:
            self.pci_slot = device_path.split("/")[-1].split("-")[1]
        else:
            self.pci_slot = None
        drm_fd = os.open(self.path, os.O_RDWR)
        amdgpu_cy.amdgpu_device_initialize(drm_fd, &major_ver, &minor_ver, &self.device_handle)
        os.close(drm_fd)
        self.memory_info = {}
        if not amdgpu_cy.amdgpu_query_info(self.device_handle, amdgpu_cy.AMDGPU_INFO_VRAM_GTT,
                                           sizeof(vram_gtt), &vram_gtt): # amdgpu vram gtt query succeeded
            self.memory_info["vram_size"] = vram_gtt.vram_size
            self.memory_info["gtt_size"] = vram_gtt.gtt_size
            self.memory_info["vram_cpu_accessible_size"] = vram_gtt.vram_cpu_accessible_size
        else:
            self.memory_info["vram_size"] = None
            self.memory_info["gtt_size"] = None
            self.memory_info["vram_cpu_accessible_size"] = None

        device_name = amdgpu_cy.amdgpu_get_marketing_name(self.device_handle)
        # null check device name (segfault gang)
        if device_name:
            self.name = device_name.decode()
        else:
            self.name = None

    cpdef object start_utilisation_polling(self, int ticks_per_second=100, int buffer_size_in_ticks=100):
        """Starts the utilisation polling thread

        Params:
            ticks_per_second (optional, default 100): int; specifies the number of device polls
            that will be performed each second.
            buffer_size_in_ticks (optional, default 100): int; specifies the number of ticks
            which will be stored.
        Raises:
            RuntimeError: if the thread is already running.
            OSError: if the polling thread fails to start.
        Returns:
            Nothing returned.
        Extra information:
            - This function starts a thread which reads GPU performance registers {ticks_per_second} times per second
                - These registers store the status of GPU components, simply as 1 (in use) or 0 (not in use) for each component
            - The results of these reads is saved to a buffer of {buffer_size_in_ticks}
            - When utilisation is queried, a mean is taken of all the values in the buffer
                - e.g texture addresser data may be 0000100100, 10 samples with 2 ones, so utilisation is 0.2 
            - This means that the time period of the mean is buffer_size_in_ticks / ticks_per_second
                - So at default it is a mean of the past second (100/100 = 1)
            - When the thread has just started, querying utilisation before one time period has elasped will error
                - The buffer will not have been populated at this point and will be partly zeroes, so a mean would be misleading
            - The polling thread provides data only for query_utilisation

            WARNING: Utilisation polling may not work correctly with all devices, in testing it
            has been unreliable with some APUs.
        """
        cdef:
            pthread_cy.pthread_t tid
            pthread_cy.pthread_attr_t attr
            int index
        if self.utilisation_polling:
            raise RuntimeError("Thread already started")
        pthread_cy.pthread_attr_init(&attr)
        pthread_cy.pthread_attr_setdetachstate(&attr, pthread_cy.PTHREAD_CREATE_DETACHED)
        self.thread_args = <poll_args_t*>malloc(sizeof(poll_args_t))
        self.thread_args.desired_state = 1
        self.thread_args.current_state = 0
        self.thread_args.measurement_interval.tv_nsec = int((1 / ticks_per_second * 1e9) % 1e9)
        self.thread_args.measurement_interval.tv_sec = 1 // ticks_per_second
        self.thread_args.buffer_size = buffer_size_in_ticks
        self.thread_args.results = <uint32_t*>calloc(buffer_size_in_ticks, sizeof(uint32_t))
        self.thread_args.device_handle = &self.device_handle
        self.utilisation_polling = True
        if pthread_cy.pthread_create(&tid, &attr, poll_registers, self.thread_args) != 0:
            # cleans up
            self.stop_utilisation_polling()
            raise OSError("Poll thread failed to start")

    cpdef object stop_utilisation_polling(self):
        """Stops the utilisation polling thread

        Params:
            No params.
        Raises:
            RuntimeError: if the thread was does not exist.
        Returns:
            Nothing returned.
        Extra information:
            This also frees all resources allocated for the thread.
        """
        if not self.utilisation_polling:
            raise RuntimeError("Thread non-existent (never started or already stopped)")
        self.thread_args.desired_state = 0
        # wait for it to exit
        while self.thread_args.current_state != 0:
            time.sleep(0.001)
        free(self.thread_args.results)
        free(self.thread_args)
        self.utilisation_polling = False

    cpdef object query_utilisation(self):
        """Queries utilisation of different GPU units
        
        Params:
            No params.
        Raises:
            RuntimeError: details of the error will be in the error message (see extra infomation).
        Returns:
            dict containing utilisations from 0 - 1 (float):
                - "texture_addresser"
                - "shader_export"
                - "shader_interpolator"
                - "scan_converter"
                - "primitive_assembly"
                - "depth_block"
                - "colour_block"
                - "graphics_pipe"
        Extra information:
            RuntimeError can be raised for:
                - Utilisation polling thread not existing/alive if utilisation requested
                - Utilisation polling thread initialising
        """
        cdef:
            dict utilisation
            int index
            uint32_t gbrm_value
            str bit
            unsigned int compare
        if not self.utilisation_polling:
            raise RuntimeError("Utilisation polling thread not running, call "
                               "start_utilisation_polling() (and wait for thread initialisation) "
                               "before querying utilisation")
        if self.thread_args.current_state == 0:
            raise RuntimeError("Utilisation polling thread is not alive")
        if self.thread_args.current_state == 2:
            raise RuntimeError("Utilisation polling thread initialising, "
                               "this takes {0:.2f}s with current options (see start_utilisation_polling docstring)"
                                .format(self.thread_args.buffer_size
                                        * (self.thread_args.measurement_interval.tv_nsec / 1e9
                                           + self.thread_args.measurement_interval.tv_sec)))
        utilisation = {k: 0 for k in COMPARE_BITS}
        for index in range(self.thread_args.buffer_size):
            gbrm_value = self.thread_args.results[index]
            for bit, compare in COMPARE_BITS.items():
                if gbrm_value & compare:
                    utilisation[bit] += 1
        return {k: v / self.thread_args.buffer_size for k, v in utilisation.items()}

    cdef object check_amdgpu_retcode(self, int retcode):
        if retcode != 0:
            raise RuntimeError("Unknown query failure (an amdgpu call failed). This query may not be supported for this GPU.")

    cpdef object query_max_clocks(self):
        """Queries max GPU clocks
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            dict,
                - "max_sclk": int, hertz
                - "max_mclk": int, hertz
        """
        cdef:
            int multiplier = 1000
            amdgpu_cy.amdgpu_gpu_info gpu_info_struct
        self.check_amdgpu_retcode(amdgpu_cy.amdgpu_query_gpu_info(self.device_handle, &gpu_info_struct))
        return {k: v * multiplier for k, v in dict(sclk_max=gpu_info_struct.max_engine_clk,
                                                   mclk_max=gpu_info_struct.max_memory_clk).items()}

    cpdef object query_vram_usage(self):
        """Queries VRAM usage
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            int, usage in bytes.
        """
        cdef:
            unsigned long long out = 0
            int multiplier = 1
        self.check_amdgpu_retcode(query_info(self.device_handle, amdgpu_cy.AMDGPU_INFO_VRAM_USAGE, &out))
        return out * multiplier

    cpdef object query_gtt_usage(self):
        """Queries GTT usage
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            int, usage in bytes.
        """
        cdef:
            unsigned long long out = 0
            int multiplier = 1
        self.check_amdgpu_retcode(query_info(self.device_handle, amdgpu_cy.AMDGPU_INFO_GTT_USAGE, &out))
        return out * multiplier
    
    cpdef object query_sclk(self):
        """Queries shader (core) clock
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            int, shader clock in hertz.
        """
        cdef:
            unsigned long long out = 0
            int multiplier = 1000000
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_SCLK, &out))
        return out * multiplier
    
    cpdef object query_mclk(self):
        """Queries memory clock
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            int, memory clock in hertz.
        """
        cdef:
            unsigned long long out = 0
            int multiplier = 1000000
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_GFX_MCLK, &out))
        return out * multiplier

    cpdef object query_temperature(self):
        """Queries temperature
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            float, temperature in °C.
        """
        cdef:
            unsigned long long out = 0
            double multiplier = 0.001
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_TEMP, &out))
        return out * multiplier

    cpdef object query_load(self):
        """Queries overall GPU load
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            float, value between 0 and 1.
        """
        cdef:
            unsigned long long out = 0
            double multiplier = 0.01
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_LOAD, &out))
        return out * multiplier

    cpdef object query_power(self):
        """Queries power consumption
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            int, consumption in W.
        """
        cdef:
            unsigned long long out = 0
            int multiplier = 1
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_GPU_AVG_POWER, &out))
        return out * multiplier

    cpdef object query_northbridge_voltage(self):
        """Queries northbridge voltage
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            float, voltage in V.
        """
        cdef:
            unsigned long long out = 0
            double multiplier = 0.001
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_VDDNB, &out))
        return out * multiplier
    
    cpdef object query_graphics_voltage(self):
        """Queries graphics voltage
        
        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling amdgpu methods.
        Returns:
            float, voltage in V.
        """
        cdef:
            unsigned long long out = 0
            double multiplier = 0.001
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_VDDGFX, &out))
        return out * multiplier
    

    # THESE ARE UNSUPPORTED AS OF NOW DUE TO THESE REQUIRING RECENT libdrm
    # THESE ARE ALSO UNTESTED AND PROBABLY DO NOT WORK


    # cpdef object query_sclk_pstate(self):
    #     """Queries shader (core) pstate
        
    #     Params:
    #         No params.
    #     Raises:
    #         RuntimeError: if an error occurs when calling amdgpu methods.
    #     Returns:
    #         int, pstate.
    #     """
    #     cdef:
    #         unsigned long long out = 0
    #         int multiplier = 1
    #     self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_SCLK, &out))
    #     return out * multiplier

    # cpdef object query_mclk_pstate(self):
    #     """Queries memory pstate
        
    #     Params:
    #         No params.
    #     Raises:
    #         RuntimeError: if an error occurs when calling amdgpu methods.
    #     Returns:
    #         int, pstate.
    #     """
    #     cdef:
    #         unsigned long long out = 0
    #         int multiplier = 1
    #     self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_MCLK, &out))
    #     return out * multiplier
    