/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef L0_HELPERS_H
#define L0_HELPERS_H

#include <cassert>
#include <fstream>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <sstream>
#include <vector>

#include "common_helpers.h"

#define L0_SAFE_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            fprintf(stderr, "%s:%d: L0 error 0x%x\n", __FILE__, __LINE__, (int)status);                                \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

namespace hostutil {
void L0InitContext(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_context_handle_t &hContext,
                   ze_module_handle_t &hModule, ze_command_queue_handle_t &hCommandQueue, const char *filename,
                   bool use_zebin) {
    L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Retrieve drivers
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));
    assert(driverCount != 0);

    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers.data()));

    // Find an instance of Intel GPU device
    // User can select particular device using env variable
    // By default first available device is selected
    auto gpuDeviceToGrab = 0;
    const char *gpuDeviceEnv = getenv("ISPC_GPU_DEVICE");
    if (gpuDeviceEnv) {
        std::istringstream(gpuDeviceEnv) >> gpuDeviceToGrab;
    } else {
        // Allow using ISPCRT env to make things easier
        const char *gpuDeviceEnv = getenv("ISPCRT_GPU_DEVICE");
        if (gpuDeviceEnv) {
            std::istringstream(gpuDeviceEnv) >> gpuDeviceToGrab;
        }
    }

    auto gpuDevice = 0;
    for (auto &driver : allDrivers) {
        uint32_t deviceCount = 0;
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, nullptr));
        std::vector<ze_device_handle_t> allDevices(deviceCount);
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, allDevices.data()));

        for (auto &device : allDevices) {
            ze_device_properties_t device_properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            L0_SAFE_CALL(zeDeviceGetProperties(device, &device_properties));
            if (device_properties.type == ZE_DEVICE_TYPE_GPU && device_properties.vendorId == 0x8086) {
                gpuDevice++;
                if (gpuDevice == gpuDeviceToGrab + 1) {
                    hDevice = device;
                    hDriver = driver;
                    break;
                }
            }
        }

        if (hDevice)
            break;
    }

    assert(hDriver);
    assert(hDevice);

    // Create a context
    ze_context_desc_t contextDesc = {}; // use default values
    L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));

    // Create a command queue
    ze_command_queue_desc_t commandQueueDesc = {};
    commandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    commandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    L0_SAFE_CALL(zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

    std::ifstream ins;
    std::string fn = filename;
    fn += (use_zebin ? ".bin" : ".spv");
    ins.open(fn, std::ios::binary);
    if (!ins.good()) {
        fprintf(stderr, "Open %s failed\n", fn.c_str());
        return;
    }

    ins.seekg(0, std::ios::end);
    size_t codeSize = ins.tellg();
    ins.seekg(0, std::ios::beg);

    if (codeSize == 0) {
        return;
    }

    unsigned char *codeBin = new unsigned char[codeSize];
    if (!codeBin) {
        return;
    }

    ins.read((char *)codeBin, codeSize);
    ins.close();

    ze_module_desc_t moduleDesc = {};
    moduleDesc.format = use_zebin ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = codeSize;
    moduleDesc.pInputModule = codeBin;
    moduleDesc.pBuildFlags = "-vc-codegen -no-optimize -Xfinalizer '-presched' -Xfinalizer '-newspillcostispc'";
    L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));

    delete[] codeBin;
}

void L0DestroyContext(ze_driver_handle_t hDriver, ze_device_handle_t hDevice, ze_context_handle_t hContext,
                      ze_module_handle_t hModule, ze_command_queue_handle_t hCommandQueue) {
    if (hCommandQueue)
        L0_SAFE_CALL(zeCommandQueueDestroy(hCommandQueue));
    if (hModule)
        L0_SAFE_CALL(zeModuleDestroy(hModule));
    if (hContext)
        L0_SAFE_CALL(zeContextDestroy(hContext));
}

void L0Create_Kernel(ze_device_handle_t &hDevice, ze_context_handle_t &hContext, ze_module_handle_t &hModule,
                     ze_command_list_handle_t &hCommandList, ze_kernel_handle_t &hKernel, const char *name) {
    // Create a command list
    ze_command_list_desc_t commandListDesc = {};
    L0_SAFE_CALL(zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList));
    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.pKernelName = name;
    L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));

    // Set device/shared indirect flags
    ze_kernel_indirect_access_flags_t kernel_flags =
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE | ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    L0_SAFE_CALL(zeKernelSetIndirectAccess(hKernel, kernel_flags));
}

void L0Destroy_Kernel(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel) {
    if (hKernel)
        L0_SAFE_CALL(zeKernelDestroy(hKernel));
    if (hCommandList)
        L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

void L0Create_EventPool(ze_device_handle_t hDevice, ze_context_handle_t hContext, const size_t size,
                        ze_event_pool_handle_t &hPool) {
    // Create event pool and enable time measurements
    ze_event_pool_desc_t eventPoolDesc = {};
    eventPoolDesc.count = size;
    eventPoolDesc.flags = (ze_event_pool_flag_t)(ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
    L0_SAFE_CALL(zeEventPoolCreate(hContext, &eventPoolDesc, 1, &hDevice, &hPool));
}

void L0Destroy_EventPool(ze_event_pool_handle_t hPool) { L0_SAFE_CALL(zeEventPoolDestroy(hPool)); }

}; // namespace hostutil

#endif
