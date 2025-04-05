"""GPU information using AMDGPU"""
cimport pyamdgpuinfo.xf86drm_cy as xf86drm_cy
cimport pyamdgpuinfo.amdgpu_cy as amdgpu_cy
cimport pyamdgpuinfo.pthread_cy as pthread_cy

from libc.stdint cimport uint32_t
from libc.stdlib cimport malloc, calloc, free
from posix.time cimport CLOCK_MONOTONIC, timespec, clock_gettime, nanosleep

import os


# gpu registers location
cdef int GRBM_OFFSET = 8196
# search path
cdef str DEVICE_PATH = "/dev/dri/by-path/"
cdef str DEVICE_FALLBACK_PATH = "/dev/dri/"

cdef dict COMPARE_BITS = {
    "texture_addresser": 2 ** 14,
    "shader_export": 2 ** 20,
    "shader_interpolator": 2 ** 22,
    "scan_converter": 2 ** 24,
    "primitive_assembly": 2 ** 25,
    "depth_block": 2 ** 26,
    "colour_block": 2 ** 30,
    "graphics_pipe": 2 ** 31
}

cdef enum UtilisationThreadState:
    OFF = 0
    RUN = 1
    WARMUP = 2


cdef int get_registers(amdgpu_cy.amdgpu_device_handle amdgpu_dev, uint32_t *out) noexcept nogil:
    """Returns whether AMDGPU query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_read_mm_registers(amdgpu_dev, GRBM_OFFSET, 1, 0xffffffff, 0, out)

cdef int query_info(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out) noexcept nogil:
    """Returns whether AMDGPU query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_info(amdgpu_dev, info, sizeof(unsigned long long), out)

cdef int query_sensor(amdgpu_cy.amdgpu_device_handle amdgpu_dev, int info, unsigned long long *out) noexcept nogil:
    """Returns whether AMDGPU query succeeds, returns query result through out"""
    return amdgpu_cy.amdgpu_query_sensor_info(amdgpu_dev, info, sizeof(unsigned long long), out)


cdef timespec timespec_subtract(timespec left, timespec right) noexcept nogil:
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

cdef timespec timespec_clamp(timespec time_to_clamp) noexcept nogil:
    """Clamps a timespec to 0 if seconds is negative"""
    if time_to_clamp.tv_sec < 0:
        time_to_clamp.tv_sec = 0
        time_to_clamp.tv_nsec = 0
    return time_to_clamp

cdef timespec timespec_sum(timespec left, timespec right) noexcept nogil:
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
    UtilisationThreadState desired_state
    # the thread's actual state
    UtilisationThreadState current_state
    timespec measurement_interval
    int buffer_size
    uint32_t* results
    amdgpu_cy.amdgpu_device_handle* device_handle


cdef void* poll_registers(void *arg) noexcept nogil:
    cdef:
        int index = 0
        poll_args_t *args
        uint32_t register_data
        timespec iteration_start_time
        timespec iteration_end_time
        timespec sleep_start_time
        timespec sleep_duration
    args = <poll_args_t*>arg
    args.current_state = UtilisationThreadState.WARMUP
    # timer init
    sleep_duration = args.measurement_interval
    if clock_gettime(CLOCK_MONOTONIC, &sleep_start_time) != 0:
        args.current_state = UtilisationThreadState.OFF
        return NULL
    sleep_start_time = timespec_subtract(sleep_start_time, sleep_duration)
    while args.desired_state == UtilisationThreadState.RUN:
        clock_gettime(CLOCK_MONOTONIC, &iteration_start_time)
        if get_registers(args.device_handle[0], &register_data) != 0:
            args.current_state = UtilisationThreadState.OFF
            return NULL
        args.results[index] = register_data
        index += 1
        if index >= args.buffer_size:
            args.current_state = UtilisationThreadState.RUN
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
    args.current_state = UtilisationThreadState.OFF
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
        OSError: if an AMDGPU device couldn't be initialised.
    Returns:
        int; number of GPUs available.
    """
    return get_gpu(-1)


cpdef object get_gpu(int gpu_id):
    """Returns GPUInfo object for gpu_id

    Params:
        gpu_id: int; the sequential id of the GPU to get.
    Raises:
        OSError: if an AMDGPU device couldn't be initialised.
        RuntimeError: if the GPU requested could not be found.
    Returns:
        GPUInfo; object providing interface to GPU info calls.
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
        int amdgpu_index = 0
        str gpu_path
        str drm_name
    devices = find_devices()
    for gpu_path in devices:
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
            raise OSError("Can't initialize AMDGPU driver for AMDGPU GPU"
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
    raise RuntimeError(f"GPU id {gpu_id} not found")


cdef class GPUInfo:
    """Interface for AMDGPU device info queries

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
        start_utilisation_polling: starts utilisation polling.
        stop_utilisation_polling: stops utilisation polling.
        query_utilisation: calculates utilisation of different GPU blocks (relies on above polling).

        query_max_clocks: queries max GPU clocks.
        query_vram_usage: queries VRAM usage.
        query_gtt_usage: queries GTT usage.
        query_sclk: queries shader (core) clock.
        query_mclk: queries memory clock.
        query_temperature: queries temperature.
        query_load: queries overall GPU load.
        query_power: queries power consumption.
        query_northbridge_voltage: queries northbridge voltage.
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
        cdef:
            timespec sleep_time
        sleep_time.tv_nsec = 1_000_000
        sleep_time.tv_sec = 0
        if self.utilisation_polling:
            self.thread_args.desired_state = UtilisationThreadState.OFF
            # wait for it to exit
            while self.thread_args.current_state != UtilisationThreadState.OFF:
                nanosleep(&sleep_time, NULL)
            free(self.thread_args.results)
            free(self.thread_args)
            self.utilisation_polling = False
        amdgpu_cy.amdgpu_device_deinitialize(self.device_handle)

    def __init__(self, device_path, gpu_id):
        cdef:
            amdgpu_cy.drm_amdgpu_info_vram_gtt vram_gtt
            const char* device_name
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
                                           sizeof(vram_gtt), &vram_gtt): # AMDGPU vram gtt query succeeded
            self.memory_info["vram_size"] = vram_gtt.vram_size
            self.memory_info["gtt_size"] = vram_gtt.gtt_size
            self.memory_info["vram_cpu_accessible_size"] = vram_gtt.vram_cpu_accessible_size
        else:
            self.memory_info["vram_size"] = None
            self.memory_info["gtt_size"] = None
            self.memory_info["vram_cpu_accessible_size"] = None

        device_name = amdgpu_cy.amdgpu_get_marketing_name(self.device_handle)
        # device name can be null
        if device_name:
            self.name = device_name.decode()
        else:
            self.name = None

    cpdef object start_utilisation_polling(self, int polls_per_second=100, int averaging_period=100):
        """Starts utilisation polling used for query_utilisation()

        Params:
            polls_per_second (optional, default 100): int; the number of device polls
            that will be performed each second.
            averaging_period (optional, default 100): int; the number of polls
            to average over.
        Raises:
            RuntimeError: if polling is already running.
            OSError: if polling fails to start.
        Returns:
            Nothing returned.
        Extra information:
            - Polling provides data only for query_utilisation, other queries do not need it
                - Polling actively uses CPU time so only start polling if you need it!
            - GPU functional block activity (either active or inactive) is polled n times per second
            - When utilisation is queried, a mean is taken of all the previous values within the averaging period
                - e.g texture addresser activity may be 0000100100, 10 samples with 2 ones, so utilisation is 0.2
                - (here the averaging period is 10)
            - Therefore the time period of the mean is polls_per_second / averaging_period
                - So at defaults a mean of the past second (100/100 = 1)
            - When polling has just started, querying utilisation before one time period has elapsed will error
                - The buffer will not have been populated at this point

            WARNING: Utilisation polling may not work correctly with all devices, in testing it
            has been unreliable with some APUs.
        """
        cdef:
            pthread_cy.pthread_t tid
            pthread_cy.pthread_attr_t attr
            int index
        if self.utilisation_polling:
            raise RuntimeError("Polling already started")
        pthread_cy.pthread_attr_init(&attr)
        # why detached?
        pthread_cy.pthread_attr_setdetachstate(&attr, pthread_cy.PTHREAD_CREATE_DETACHED)
        self.thread_args = <poll_args_t*>malloc(sizeof(poll_args_t))
        if not self.thread_args:
            raise MemoryError()
        self.thread_args.desired_state = UtilisationThreadState.RUN
        self.thread_args.current_state = UtilisationThreadState.OFF
        self.thread_args.measurement_interval.tv_nsec = int((1 / polls_per_second * 1e9) % 1e9)
        self.thread_args.measurement_interval.tv_sec = 1 // polls_per_second
        self.thread_args.buffer_size = averaging_period
        self.thread_args.results = <uint32_t*>calloc(averaging_period, sizeof(uint32_t))
        if not self.thread_args.results:
            raise MemoryError()
        self.thread_args.device_handle = &self.device_handle
        self.utilisation_polling = True
        if pthread_cy.pthread_create(&tid, &attr, poll_registers, self.thread_args) != 0:
            # cleans up
            self.stop_utilisation_polling()
            raise OSError("Failed to start polling")

    cpdef object stop_utilisation_polling(self):
        """Stops utilisation polling

        Params:
            No params.
        Raises:
            No specific exceptions.
        Returns:
            Nothing returned.
        Extra information:
            This is safe to call even if utilisation polling is not currently running.
            This frees all resources allocated.
        """
        cdef:
            timespec sleep_time
        sleep_time.tv_nsec = 1_000_000
        sleep_time.tv_sec = 0

        if not self.utilisation_polling:
            return
        self.thread_args.desired_state = UtilisationThreadState.OFF
        # wait for it to exit
        while self.thread_args.current_state != UtilisationThreadState.OFF:
            nanosleep(&sleep_time, NULL)
        free(self.thread_args.results)
        free(self.thread_args)
        self.utilisation_polling = False

    cpdef object query_utilisation(self):
        """Queries utilisation of different GPU units

        Params:
            No params.
        Raises:
            RuntimeError: details of the error will be in the error message (see extra information).
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
                - Utilisation polling not started
                - Utilisation polling stopped due to an error
                - Utilisation polling initialising
        """
        cdef:
            dict utilisation
            int index
            uint32_t gbrm_value
            str bit
            unsigned int compare
        if not self.utilisation_polling:
            raise RuntimeError("Utilisation polling not started, call "
                               "start_utilisation_polling() (and wait for initialisation) "
                               "before querying utilisation")
        if self.thread_args.current_state == UtilisationThreadState.OFF:
            raise RuntimeError("Utilisation polling exited due to an error")
        if self.thread_args.current_state == UtilisationThreadState.WARMUP:
            raise RuntimeError("Utilisation polling initialising, "
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
            raise RuntimeError("Unknown query failure (an AMDGPU call failed). This query may not be supported for this GPU")

    cpdef object query_max_clocks(self):
        """Queries max GPU clocks

        Params:
            No params.
        Raises:
            RuntimeError: if an error occurs when calling into AMDGPU.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            int; usage in bytes.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            int; usage in bytes.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            int; shader clock in hertz.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            int; memory clock in hertz.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            float; temperature in °C.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            float; value between 0 and 1.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            int; consumption in W.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            float; voltage in V.
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
            RuntimeError: if an error occurs when calling into AMDGPU.
        Returns:
            float; voltage in V.
        """
        cdef:
            unsigned long long out = 0
            double multiplier = 0.001
        self.check_amdgpu_retcode(query_sensor(self.device_handle, amdgpu_cy.AMDGPU_INFO_SENSOR_VDDGFX, &out))
        return out * multiplier
