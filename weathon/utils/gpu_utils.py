# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:27
# @Author  : LiZhen
# @FileName: gpu_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import platform


class GPUUtils:
    def __init__(self, id, uuid, load, memory_total, memory_used, memory_free, driver, gpu_name, serial, display_mode,
                 display_active, temp_gpu):
        self.id = id
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memory_used) / float(memory_total)
        self.memoryTotal = memory_total
        self.memoryUsed = memory_used
        self.memoryFree = memory_free
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu
    @staticmethod
    def safeFloatCast(str_number):
        try:
            number = float(str_number)
        except ValueError:
            number = float('nan')
        return number

    @staticmethod
    def getGPUs():
        if platform.system() == "Windows":
            # If the platform is Windows and nvidia-smi
            # could not be found from the environment path,
            # try to find it from system drive with default installation path
            nvidia_smi = spawn.find_executable('nvidia-smi')
            if nvidia_smi is None:
                nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
        else:
            nvidia_smi = "nvidia-smi"

        # Get ID, processing and memory utilization for all GPUs
        try:
            p = Popen([nvidia_smi,
                       "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                       "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, stderror = p.communicate()
        except:
            return []
        output = stdout.decode('UTF-8')
        # output = output[2:-1] # Remove b' and ' from string added by python
        # print(output)

        # Split on line break
        lines = output.split(os.linesep)
        num_devices = len(lines) - 1
        gpus = []
        for g in range(num_devices):
            line = lines[g]
            # print(line)
            vals = line.split(', ')
            # print(vals)
            for i in range(12):
                # print(vals[i])
                if i == 0:
                    device_ids = int(vals[i])
                elif i == 1:
                    uuid = vals[i]
                elif i == 2:
                    gpu_util = GPUUtils.safeFloatCast(vals[i]) / 100
                elif i == 3:
                    mem_total = GPUUtils.safeFloatCast(vals[i])
                elif i == 4:
                    mem_used = GPUUtils.safeFloatCast(vals[i])
                elif i == 5:
                    mem_free = GPUUtils.safeFloatCast(vals[i])
                elif i == 6:
                    driver = vals[i]
                elif i == 7:
                    gpu_name = vals[i]
                elif i == 8:
                    serial = vals[i]
                elif i == 9:
                    display_active = vals[i]
                elif i == 10:
                    display_mode = vals[i]
                elif i == 11:
                    temp_gpu = GPUUtils.safeFloatCast(vals[i]);

            gpus.append(
                GPUUtils(device_ids, uuid, gpu_util, mem_total, mem_used, mem_free, driver, gpu_name, serial, display_mode,
                    display_active, temp_gpu))
        return gpus  # (deviceIds, gpuUtil, memUtil)

    @classmethod
    def getAvailabilityGPU(cls, gpus, max_load=0.5, max_memory=0.5, memory_free=0, include_nan=False, exclude_id=[],
                           exclude_uuid=[]):
        # Determine, which GPUs are available
        gpu_availability = [
            1 if (gpu.memoryFree >= memory_free) and (
                        gpu.load < max_load or (include_nan and math.isnan(gpu.load))) and (
                         gpu.memoryUtil < max_memory or (include_nan and math.isnan(gpu.memoryUtil))) and (
                         (gpu.id not in exclude_id) and (gpu.uuid not in exclude_uuid)) else 0 for gpu in gpus]
        gpus = [gpu for gpu, use in zip(gpus, gpu_availability) if use]
        return gpus


