/*
  Copyright (c) 2019-2020, Intel Corporation
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

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#endif // ISPC_IS_WINDOWS

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif

/******************************/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <level_zero/ze_api.h>
#include <limits>
#include <math.h>
#include <sstream>
#include <string>

#define L0_SAFE_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            fprintf(stderr, "%s:%d: L0 error %d\n", __FILE__, __LINE__, (int)status);                                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#define N 64

int width() {
#if defined(TEST_WIDTH)
    return TEST_WIDTH;
#else
#error "Unknown or unset TEST_WIDTH value"
#endif
}

#if defined(_WIN32) || defined(_WIN64)
#define ALIGN
#else
#define ALIGN __attribute__((aligned(64)))
#endif

static void L0InitContext(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue) {
    L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_NONE));

    // Discover all the driver instances
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

    ze_driver_handle_t *allDrivers = (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
    L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

    // Find a driver instance with a GPU device
    for (uint32_t i = 0; i < driverCount; ++i) {
        uint32_t deviceCount = 0;
        hDriver = allDrivers[i];
        L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
        ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
        L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
        for (uint32_t d = 0; d < deviceCount; ++d) {
            ze_device_properties_t device_properties;
            L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                hDevice = allDevices[d];
                break;
            }
        }
        free(allDevices);
        if (nullptr != hDevice) {
            break;
        }
    }
    free(allDrivers);
    assert(hDriver);
    assert(hDevice);
    // Create a command queue
    ze_command_queue_desc_t commandQueueDesc = {ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT, ZE_COMMAND_QUEUE_FLAG_NONE,
                                                ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0};
    L0_SAFE_CALL(zeCommandQueueCreate(hDevice, &commandQueueDesc, &hCommandQueue));

    std::ifstream is;
    std::string fn = "test_genx.spv";
    is.open(fn, std::ios::binary);
    if (!is.good()) {
        fprintf(stderr, "Open %s failed\n", fn.c_str());
        return;
    }

    is.seekg(0, std::ios::end);
    size_t codeSize = is.tellg();
    is.seekg(0, std::ios::beg);

    if (codeSize == 0) {
        return;
    }

    unsigned char *codeBin = new unsigned char[codeSize];
    if (!codeBin) {
        return;
    }

    is.read((char *)codeBin, codeSize);
    is.close();

    ze_module_desc_t moduleDesc = {ZE_MODULE_DESC_VERSION_CURRENT, //
                                   ZE_MODULE_FORMAT_IL_SPIRV,      //
                                   codeSize,                       //
                                   codeBin,                        //
                                   "-vc-codegen -no-optimize"};
    L0_SAFE_CALL(zeModuleCreate(hDevice, &moduleDesc, &hModule, nullptr));
}

static void L0Create_Kernel(ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                            ze_command_list_handle_t &hCommandList, ze_kernel_handle_t &hKernel, const char *name) {
    // Create a command list
    ze_command_list_desc_t commandListDesc = {ZE_COMMAND_LIST_DESC_VERSION_CURRENT, ZE_COMMAND_LIST_FLAG_NONE};
    L0_SAFE_CALL(zeCommandListCreate(hDevice, &commandListDesc, &hCommandList));
    ze_kernel_desc_t kernelDesc = {ZE_KERNEL_DESC_VERSION_CURRENT, //
                                   ZE_KERNEL_FLAG_NONE,            //
                                   name};

    L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));
}

static void L0Launch_Kernel(ze_command_queue_handle_t &hCommandQueue, ze_command_list_handle_t &hCommandList,
                            ze_kernel_handle_t &hKernel, int bufsize = 0, void *return_data = nullptr,
                            void *OUTBuff = nullptr, int groupSpaceWidth = 1, int groupSpaceHeight = 1) {
    // set group size
    uint32_t group_size = groupSpaceWidth * groupSpaceHeight;
    L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ groupSpaceWidth, /*y*/ groupSpaceHeight, /*z*/ 1));

    // set grid size
    ze_group_count_t dispatchTraits = {1, 1, 1};

    // launch
    L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));

    L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

    // copy result to host
    if (return_data && OUTBuff)
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, return_data, OUTBuff, bufsize, nullptr));
    // dispatch & wait
    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, (std::numeric_limits<uint32_t>::max)()));
}

static void L0Launch_F_V(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                         ze_command_queue_handle_t &hCommandQueue, void *return_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_v");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_Threads(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                               ze_command_queue_handle_t &hCommandQueue, void *return_data, int groupSpaceWidth,
                               int groupSpaceHeight) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_t");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff, groupSpaceWidth,
                    groupSpaceHeight);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_F(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                         ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_f");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_FI(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data,
                          void *vint_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_fi");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr, *IN1Buff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &INBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(int), N * sizeof(int), hDevice, &IN1Buff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, IN1Buff, vint_data, N * sizeof(int), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(IN1Buff), &IN1Buff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, IN1Buff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_FU(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_fu");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DU(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data, double b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_du");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(double), N * sizeof(double), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(double), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DUF(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                           ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_duf");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(double), N * sizeof(double), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DI(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data,
                          void *vint2_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "f_di");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr, *INBuff = nullptr, *IN1Buff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(double), N * sizeof(double), hDevice, &INBuff));
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(int), N * sizeof(int), hDevice, &IN1Buff));

    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, IN1Buff, vint2_data, N * sizeof(int), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(IN1Buff), &IN1Buff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, IN1Buff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_UF(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                              ze_command_queue_handle_t &hCommandQueue, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "print_uf");

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_F(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                             ze_command_queue_handle_t &hCommandQueue, void *vfloat_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "print_f");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(INBuff), &INBuff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_FUF(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                               ze_command_queue_handle_t &hCommandQueue, void *vfloat_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "print_fuf");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *INBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeDriverFreeMem(hDriver, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_NO(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                              ze_command_queue_handle_t &hCommandQueue) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "print_no");

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Result(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                            ze_command_queue_handle_t &hCommandQueue, void *return_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "result");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_Result(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                                  ze_command_queue_handle_t &hCommandQueue) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "print_result");

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Result_Threads(ze_driver_handle_t &hDriver, ze_device_handle_t &hDevice,
                                    ze_module_handle_t &hModule, ze_command_queue_handle_t &hCommandQueue,
                                    void *return_data, int groupSpaceWidth, int groupSpaceHeight) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "result_t");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, N * sizeof(float), N * sizeof(float), hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr));
    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(int), &groupSpaceWidth));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(int), &groupSpaceHeight));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff, groupSpaceWidth,
                    groupSpaceHeight);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, OUTBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

int main(int argc, char *argv[]) {
    // init data
    struct alignas(4096) AlignedArray {
        float data[N];
    } returned_result, expected_result, vfloat;
    struct alignas(4096) AlignedArray1 {
        int data[N];
    } vint, vint2;
    struct alignas(4096) AlignedArray2 {
        double data[N];
    } vdouble;

    for (int i = 0; i < N; ++i) {
        returned_result.data[i] = float(-1e20);
        vfloat.data[i] = float(i + 1);
        vdouble.data[i] = double(i + 1);
        vint.data[i] = 2 * (i + 1);
        vint2.data[i] = i + 5;
    }

    void *return_data = returned_result.data;
    void *expect_data = expected_result.data;
    void *vfloat_data = vfloat.data;
    void *vint_data = vint.data;
    void *vint2_data = vint2.data;
    void *vdouble_data = vdouble.data;
    ze_device_handle_t hDevice = nullptr;
    ze_module_handle_t hModule = nullptr;
    ze_driver_handle_t hDriver = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue);
#if (TEST_SIG == 0)
    L0Launch_F_V(hDriver, hDevice, hModule, hCommandQueue, return_data);
#elif (TEST_SIG == 1)
    L0Launch_F_F(hDriver, hDevice, hModule, hCommandQueue, return_data, vfloat_data);
#elif (TEST_SIG == 2)
    float num = 5.0f;
    L0Launch_F_FU(hDriver, hDevice, hModule, hCommandQueue, return_data, vfloat_data, num);
#elif (TEST_SIG == 3)
    L0Launch_F_FI(hDriver, hDevice, hModule, hCommandQueue, return_data, vfloat_data, vint_data);
#elif (TEST_SIG == 4)
    double num = 5.0;
    L0Launch_F_DU(hDriver, hDevice, hModule, hCommandQueue, return_data, vdouble_data, num);
#elif (TEST_SIG == 5)
    float num = 5.0f;
    L0Launch_F_DUF(hDriver, hDevice, hModule, hCommandQueue, return_data, vdouble_data, num);
#elif (TEST_SIG == 6)
    L0Launch_F_DI(hDriver, hDevice, hModule, hCommandQueue, return_data, vdouble_data, vint2_data);
#elif (TEST_SIG == 7)
// L0Launch_F_SZ(return_data);
#error "Currently unsupported for GEN"
#elif (TEST_SIG == 8)
    int groupSpaceWidth = 2;
    int groupSpaceHeight = 8;
    assert(N >= groupSpaceWidth * groupSpaceHeight);
    L0Launch_F_Threads(hDriver, hDevice, hModule, hCommandQueue, return_data, groupSpaceWidth, groupSpaceHeight);
    L0Launch_Result_Threads(hDriver, hDevice, hModule, hCommandQueue, expect_data, groupSpaceWidth, groupSpaceHeight);
#elif (TEST_SIG == 32)
    L0Launch_Print_UF(hDriver, hDevice, hModule, hCommandQueue, 5.0f);
#elif (TEST_SIG == 33)
    L0Launch_Print_F(hDriver, hDevice, hModule, hCommandQueue, vfloat_data);
#elif (TEST_SIG == 34)
    L0Launch_Print_FUF(hDriver, hDevice, hModule, hCommandQueue, vfloat_data, 5.0f);
#elif (TEST_SIG == 35)
    L0Launch_Print_NO(hDriver, hDevice, hModule, hCommandQueue);
#else
#error "Unknown or unset TEST_SIG value"
#endif
#if 0
        const bool verbose = true;
#else
    const bool verbose = false;
#endif
#if (TEST_SIG < 8)
    L0Launch_Result(hDriver, hDevice, hModule, hCommandQueue, expect_data);
#elif (TEST_SIG >= 32)
    L0Launch_Print_Result(hDriver, hDevice, hModule, hCommandQueue);
    return 0;
#endif
    L0_SAFE_CALL(zeModuleDestroy(hModule));
    L0_SAFE_CALL(zeCommandQueueDestroy(hCommandQueue));

    // check results.
    int errors = 0;
    for (int i = 0; i < width(); ++i) {
        if (fabs(returned_result.data[i] - expected_result.data[i]) > 16 * FLT_EPSILON) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n", argv[0], i, returned_result.data[i],
                   returned_result.data[i], expected_result.data[i], expected_result.data[i]);
            ++errors;
#endif // EXPECT_FAILURE
        }
    }

#ifdef EXPECT_FAILURE
    // Don't expect to get here
    return 0;
#else
    return errors > 0;
#endif
}
