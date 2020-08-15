/*
  Copyright (c) 2020, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef L0_HELPERS_H
#define L0_HELPERS_H

#include <cassert>
#include <fstream>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

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
                   ze_module_handle_t &hModule, ze_command_queue_handle_t &hCommandQueue, const char *filename) {
    L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Retrieve driver
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));
    L0_SAFE_CALL(zeDriverGet(&driverCount, &hDriver));

    // Retrieve device
    uint32_t deviceCount = 0;
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, &hDevice));

    assert(hDriver);
    assert(hDevice);

    // Create a contxt
    ze_context_desc_t contextDesc = {}; // use default values
    L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));

    // Create a command queue
    ze_command_queue_desc_t commandQueueDesc = {};
    commandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    commandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    L0_SAFE_CALL(zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

    std::ifstream ins;
    std::string fn = filename;
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
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = codeSize;
    moduleDesc.pInputModule = codeBin;
    moduleDesc.pBuildFlags = "-vc-codegen -no-optimize";
    L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));
}

void L0DestroyContext(ze_driver_handle_t hDriver, ze_device_handle_t hDevice, ze_context_handle_t hContext,
                      ze_module_handle_t hModule, ze_command_queue_handle_t hCommandQueue) {
    L0_SAFE_CALL(zeCommandQueueDestroy(hCommandQueue));
    L0_SAFE_CALL(zeModuleDestroy(hModule));
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
}

void L0Destroy_Kernel(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel) {
    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

void L0Create_EventPool(ze_device_handle_t hDevice, ze_context_handle_t hContext, const size_t size,
                        ze_event_pool_handle_t &hPool) {
    // Create event pool and enable time measurements
    ze_event_pool_desc_t eventPoolDesc = {};
    eventPoolDesc.count = size;
    eventPoolDesc.flags = (ze_event_pool_flag_t)(ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP);
    L0_SAFE_CALL(zeEventPoolCreate(hContext, &eventPoolDesc, 1, &hDevice, &hPool));
}

void L0Destroy_EventPool(ze_event_pool_handle_t hPool) { L0_SAFE_CALL(zeEventPoolDestroy(hPool)); }

}; // namespace hostutil

#endif
