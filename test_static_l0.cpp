/*
  Copyright (c) 2019-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#error "L0 is not supported on macOS"
#elif defined(__FreeBSD__)
#error "L0 is not supported on FreeBSD"
#else
#error "Host OS was not detected"
#endif

#ifdef ISPC_IS_WINDOWS
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
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
#include <vector>

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

static void L0InitContext(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                          ze_command_queue_handle_t &hCommandQueue) {
    L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Retrieve drivers
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

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
    ze_driver_handle_t hDriver = 0;
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

    // Create default command context
    ze_context_desc_t contextDesc = {}; // use default values
    L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));

    // Create a command queue
    ze_command_queue_desc_t commandQueueDesc = {};
    commandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    commandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    L0_SAFE_CALL(zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

    std::ifstream is;
#ifdef TEST_ZEBIN
    std::string fn = "test_xe.bin";
#else
    std::string fn = "test_xe.spv";
#endif
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

    std::string igcOptions = "-vc-codegen -no-optimize -Xfinalizer '-presched' -Xfinalizer '-newspillcostispc'";
    const char *userIgcOptionsEnv = getenv("ISPCRT_IGC_OPTIONS");
    if (userIgcOptionsEnv) {
        std::string userIgcOptions(userIgcOptionsEnv);

        if (userIgcOptions.length() >= 3) {
            auto prefix = userIgcOptions.substr(0, 2);
            if (prefix == "+ ") {
                igcOptions += ' ' + userIgcOptions.substr(2);
            } else if (prefix == "= ") {
                igcOptions = userIgcOptions.substr(2);
            } else {
                throw std::runtime_error("Invalid ISPCRT_IGC_OPTIONS string" + userIgcOptions);
            }
        } else {
            throw std::runtime_error("Invalid ISPCRT_IGC_OPTIONS string" + userIgcOptions);
        }
    }

    // Create module
    ze_module_desc_t moduleDesc = {};
#ifdef TEST_ZEBIN
    moduleDesc.format = ZE_MODULE_FORMAT_NATIVE;
#else
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
#endif
    moduleDesc.pInputModule = codeBin;
    moduleDesc.inputSize = codeSize;
    moduleDesc.pBuildFlags = igcOptions.c_str();
    // Add build log output for easier debugginer the tests
    ze_module_build_log_handle_t buildlog;
    if (zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, &buildlog) != ZE_RESULT_SUCCESS) {
        size_t szLog = 0;
        zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

        char *strLog = (char *)malloc(szLog);
        zeModuleBuildLogGetString(buildlog, &szLog, strLog);
        std::cout << "Build log:" << strLog << std::endl;

        free(strLog);
    }
    L0_SAFE_CALL(zeModuleBuildLogDestroy(buildlog));
}

static void L0Create_Kernel(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                            ze_command_list_handle_t &hCommandList, ze_kernel_handle_t &hKernel, const char *name) {
    // Create command list
    ze_command_list_desc_t commandListDesc = {};
    L0_SAFE_CALL(zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList));

    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.pKernelName = name;
    L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));

    // Set device/shared indirect flags
    ze_kernel_indirect_access_flags_t kernel_flags =
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE | ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
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
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, return_data, OUTBuff, bufsize, nullptr, 0, nullptr));
    // dispatch & wait
    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, (std::numeric_limits<uint64_t>::max)()));
}

static void L0Launch_F_V(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                         ze_command_queue_handle_t &hCommandQueue, void *return_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_v");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);
    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_Threads(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                               ze_command_queue_handle_t &hCommandQueue, void *return_data, int groupSpaceWidth,
                               int groupSpaceHeight) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_t");

    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff, groupSpaceWidth,
                    groupSpaceHeight);
    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_F(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                         ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_f");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_FI(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data,
                          void *vint_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_fi");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr, *IN1Buff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &INBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(int), 64, hDevice, &IN1Buff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, IN1Buff, vint_data, N * sizeof(int), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(IN1Buff), &IN1Buff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));
    L0_SAFE_CALL(zeMemFree(hContext, IN1Buff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_FU(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vfloat_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_fu");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DU(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data, double b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_du");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(double), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(double), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DUF(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                           ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_duf");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(double), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_F_DI(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                          ze_command_queue_handle_t &hCommandQueue, void *return_data, void *vdouble_data,
                          void *vint2_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "f_di");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr, *INBuff = nullptr, *IN1Buff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(double), 64, hDevice, &INBuff));
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(int), 64, hDevice, &IN1Buff));

    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vdouble_data, N * sizeof(double), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, IN1Buff, vint2_data, N * sizeof(int), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(IN1Buff), &IN1Buff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);

    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));
    L0_SAFE_CALL(zeMemFree(hContext, INBuff));
    L0_SAFE_CALL(zeMemFree(hContext, IN1Buff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_UF(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                              ze_command_queue_handle_t &hCommandQueue, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "print_uf");

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_F(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                             ze_command_queue_handle_t &hCommandQueue, void *vfloat_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "print_f");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(INBuff), &INBuff));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_FUF(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                               ze_command_queue_handle_t &hCommandQueue, void *vfloat_data, float b) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "print_fuf");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *INBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &INBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, INBuff, vfloat_data, N * sizeof(float), nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(INBuff), &INBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(float), &b));

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeMemFree(hContext, INBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_NO(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                              ze_command_queue_handle_t &hCommandQueue) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "print_no");

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Result(ze_device_handle_t &hDevice, ze_module_handle_t &hModule, ze_context_handle_t &hContext,
                            ze_command_queue_handle_t &hCommandQueue, void *return_data) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "result");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff);
    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Print_Result(ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                                  ze_context_handle_t &hContext, ze_command_queue_handle_t &hCommandQueue) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "print_result");

    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel);

    L0_SAFE_CALL(zeKernelDestroy(hKernel));
    L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
}

static void L0Launch_Result_Threads(ze_device_handle_t &hDevice, ze_module_handle_t &hModule,
                                    ze_context_handle_t &hContext, ze_command_queue_handle_t &hCommandQueue,
                                    void *return_data, int groupSpaceWidth, int groupSpaceHeight) {
    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hContext, hCommandList, hKernel, "result_t");
    // allocate buffers
    ze_device_mem_alloc_desc_t allocDesc = {};
    void *OUTBuff = nullptr;
    L0_SAFE_CALL(zeMemAllocDevice(hContext, &allocDesc, N * sizeof(float), 64, hDevice, &OUTBuff));
    // copy buffers to device
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(hCommandList, OUTBuff, return_data, N * sizeof(float), nullptr, 0, nullptr));
    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(OUTBuff), &OUTBuff));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(int), &groupSpaceWidth));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(int), &groupSpaceHeight));
    L0Launch_Kernel(hCommandQueue, hCommandList, hKernel, N * sizeof(float), return_data, OUTBuff, groupSpaceWidth,
                    groupSpaceHeight);
    L0_SAFE_CALL(zeMemFree(hContext, OUTBuff));

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
    ze_context_handle_t hContext = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;
    L0InitContext(hDevice, hModule, hContext, hCommandQueue);
#if (TEST_SIG == 0)
    L0Launch_F_V(hDevice, hModule, hContext, hCommandQueue, return_data);
#elif (TEST_SIG == 1)
    L0Launch_F_F(hDevice, hModule, hContext, hCommandQueue, return_data, vfloat_data);
#elif (TEST_SIG == 2)
    float num = 5.0f;
    L0Launch_F_FU(hDevice, hModule, hContext, hCommandQueue, return_data, vfloat_data, num);
#elif (TEST_SIG == 3)
    L0Launch_F_FI(hDevice, hModule, hContext, hCommandQueue, return_data, vfloat_data, vint_data);
#elif (TEST_SIG == 4)
    double num = 5.0;
    L0Launch_F_DU(hDevice, hModule, hContext, hCommandQueue, return_data, vdouble_data, num);
#elif (TEST_SIG == 5)
    float num = 5.0f;
    L0Launch_F_DUF(hDevice, hModule, hContext, hCommandQueue, return_data, vdouble_data, num);
#elif (TEST_SIG == 6)
    L0Launch_F_DI(hDevice, hModule, hContext, hCommandQueue, return_data, vdouble_data, vint2_data);
#elif (TEST_SIG == 7)
// L0Launch_F_SZ(return_data);
#error "Currently unsupported for Xe"
#elif (TEST_SIG == 8)
    int groupSpaceWidth = 2;
    int groupSpaceHeight = 16;
    assert(N >= groupSpaceWidth * groupSpaceHeight);
    L0Launch_F_Threads(hDevice, hModule, hContext, hCommandQueue, return_data, groupSpaceWidth, groupSpaceHeight);
    L0Launch_Result_Threads(hDevice, hModule, hContext, hCommandQueue, expect_data, groupSpaceWidth, groupSpaceHeight);
#elif (TEST_SIG == 32)
    L0Launch_Print_UF(hDevice, hModule, hContext, hCommandQueue, 5.0f);
#elif (TEST_SIG == 33)
    L0Launch_Print_F(hDevice, hModule, hContext, hCommandQueue, vfloat_data);
#elif (TEST_SIG == 34)
    L0Launch_Print_FUF(hDevice, hModule, hContext, hCommandQueue, vfloat_data, 5.0f);
#elif (TEST_SIG == 35)
    L0Launch_Print_NO(hDevice, hModule, hContext, hCommandQueue);
#else
#error "Unknown or unset TEST_SIG value"
#endif
#if 0
        const bool verbose = true;
#else
    const bool verbose = false;
#endif
#if (TEST_SIG < 8)
    L0Launch_Result(hDevice, hModule, hContext, hCommandQueue, expect_data);
#elif (TEST_SIG >= 32)
    L0Launch_Print_Result(hDevice, hModule, hContext, hCommandQueue);
    return 0;
#endif
    L0_SAFE_CALL(zeCommandQueueDestroy(hCommandQueue));
    L0_SAFE_CALL(zeModuleDestroy(hModule));
    L0_SAFE_CALL(zeContextDestroy(hContext));

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
