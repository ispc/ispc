// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "GPUDevice.h"

#ifdef _WIN32
#error "Windows not yet supported!"
#else
#include <dlfcn.h>
#endif
// std
#include <exception>
#include <fstream>
#include <limits>
#include <vector>
// level0
#include <level_zero/ze_api.h>

#define L0_SAFE_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            fprintf(stderr, "%s:%d: L0 error %d\n", __FILE__, __LINE__, (int)status);                                  \
        }                                                                                                              \
    }

namespace ispcrt {
namespace gpu {

struct MemoryView : public ispcrt::MemoryView {
    MemoryView(ze_driver_handle_t driver, ze_device_handle_t device, void *appMem, size_t numBytes)
        : m_hostPtr(appMem), m_size(numBytes), m_driver(driver), m_device(device) {}

    ~MemoryView() {
        if (m_devicePtr)
            L0_SAFE_CALL(zeDriverFreeMem(m_driver, m_devicePtr));
    }

    void *hostPtr() { return m_hostPtr; };

    void *devicePtr() {
        if (!m_devicePtr)
            allocate();
        return m_devicePtr;
    };

    size_t numBytes() { return m_size; };

  private:
    void allocate() {
        ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT,
                                                ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0};

        L0_SAFE_CALL(zeDriverAllocDeviceMem(m_driver, &allocDesc, m_size, m_size, m_device, &m_devicePtr));
    }

    void *m_hostPtr{nullptr};
    void *m_devicePtr{nullptr};
    size_t m_size{0};

    ze_driver_handle_t m_driver{nullptr};
    ze_device_handle_t m_device{nullptr};
};

struct Module : public ispcrt::Module {
    Module(ze_device_handle_t device, const char *moduleFile) : m_file(moduleFile) {
        std::ifstream is;
        is.open(m_file + ".spv", std::ios::binary);
        if (!is.good())
            throw std::runtime_error("Failed to open spv file!");

        is.seekg(0, std::ios::end);
        size_t codeSize = is.tellg();
        is.seekg(0, std::ios::beg);

        if (codeSize == 0)
            throw std::runtime_error("Code size is '0'!");

        m_code.resize(codeSize);

        is.read((char *)m_code.data(), codeSize);
        is.close();

        ze_module_desc_t moduleDesc = {ZE_MODULE_DESC_VERSION_CURRENT, //
                                       ZE_MODULE_FORMAT_IL_SPIRV,      //
                                       codeSize,                       //
                                       m_code.data(),                  //
                                       "-cmc"};
        L0_SAFE_CALL(zeModuleCreate(device, &moduleDesc, &m_module, nullptr));

        if (m_module == nullptr)
            throw std::runtime_error("Failed to load spv module!");
    }

    ~Module() {
        if (m_module)
            L0_SAFE_CALL(zeModuleDestroy(m_module));
    }

    ze_module_handle_t handle() const { return m_module; }

  private:
    std::string m_file;
    std::vector<unsigned char> m_code;

    ze_module_handle_t m_module{nullptr};
};

struct Kernel : public ispcrt::Kernel {
    Kernel(const ispcrt::Module &_module, const char *name) : m_fcnName(name), m_module(&_module) {
        const gpu::Module &module = (const gpu::Module &)_module;

        ze_kernel_desc_t kernelDesc = {ZE_KERNEL_DESC_VERSION_CURRENT, //
                                       ZE_KERNEL_FLAG_NONE,            //
                                       name};
        L0_SAFE_CALL(zeKernelCreate(module.handle(), &kernelDesc, &m_kernel));

        if (m_kernel == nullptr)
            throw std::runtime_error("Failed to load kernel!");

        m_module->refInc();
    }

    ~Kernel() {
        if (m_module)
            m_module->refDec();
    }

    ze_kernel_handle_t handle() const { return m_kernel; }

  private:
    std::string m_fcnName;

    const ispcrt::Module *m_module{nullptr};
    ze_kernel_handle_t m_kernel{nullptr};
};

struct TaskQueue : public ispcrt::TaskQueue {
    TaskQueue(ze_device_handle_t device) {
        ze_command_list_desc_t commandListDesc = {ZE_COMMAND_LIST_DESC_VERSION_CURRENT, ZE_COMMAND_LIST_FLAG_NONE};
        L0_SAFE_CALL(zeCommandListCreate(device, &commandListDesc, &m_cl));

        if (m_cl == nullptr)
            throw std::runtime_error("Failed to create command list!");

        ze_command_queue_desc_t commandQueueDesc = {ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT, ZE_COMMAND_QUEUE_FLAG_NONE,
                                                    ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0};

        L0_SAFE_CALL(zeCommandQueueCreate(device, &commandQueueDesc, &m_q));

        if (m_q == nullptr)
            throw std::runtime_error("Failed to create command queue!");
    }

    ~TaskQueue() {
        if (m_q)
            L0_SAFE_CALL(zeCommandQueueDestroy(m_q));
        if (m_cl)
            L0_SAFE_CALL(zeCommandListDestroy(m_cl));
    }

    void barrier() override { L0_SAFE_CALL(zeCommandListAppendBarrier(m_cl, nullptr, 0, nullptr)); }

    void copyToHost(ispcrt::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl, view.hostPtr(), view.devicePtr(), view.numBytes(), nullptr));
    }

    void copyToDevice(ispcrt::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl, view.devicePtr(), view.hostPtr(), view.numBytes(), nullptr));
    }

    void launch(ispcrt::Kernel &k, ispcrt::MemoryView *params, size_t dim0, size_t dim1, size_t dim2) override {
        auto &kernel = (gpu::Kernel &)k;

        void *param_ptr = nullptr;

        if (params)
            param_ptr = params->devicePtr();

        L0_SAFE_CALL(zeKernelSetArgumentValue(kernel.handle(), 0, sizeof(void *), &param_ptr));

        ze_group_count_t dispatchTraits = {uint32_t(dim0), uint32_t(dim1), uint32_t(dim2)};
        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_cl, kernel.handle(), &dispatchTraits, nullptr, 0, nullptr));
    }

    void sync() override {
        L0_SAFE_CALL(zeCommandListClose(m_cl));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_q, 1, &m_cl, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(m_q, std::numeric_limits<uint32_t>::max()));
        L0_SAFE_CALL(zeCommandListReset(m_cl));
    }

  private:
    ze_command_queue_handle_t m_q{nullptr};
    ze_command_list_handle_t m_cl{nullptr};
};
} // namespace gpu

GPUDevice::GPUDevice() {
    static bool initialized = false;

    if (!initialized)
        L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_NONE));

    // Discover all the driver instances
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers.data()));

    // Find a driver instance with a GPU device
    for (auto &driver : allDrivers) {
        m_driver = driver;

        uint32_t deviceCount = 0;
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, nullptr));
        std::vector<ze_device_handle_t> allDevices(deviceCount);
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, allDevices.data()));

        for (auto &device : allDevices) {
            ze_device_properties_t device_properties;
            L0_SAFE_CALL(zeDeviceGetProperties(device, &device_properties));
            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                m_device = device;
                break;
            }
        }

        if (m_device)
            break;
    }

    if (!m_device)
        throw std::runtime_error("could not find a valid GPU device");
}

MemoryView *GPUDevice::newMemoryView(void *appMem, size_t numBytes) const {
    return new gpu::MemoryView((ze_driver_handle_t)m_driver, (ze_device_handle_t)m_device, appMem, numBytes);
}

TaskQueue *GPUDevice::newTaskQueue() const { return new gpu::TaskQueue((ze_device_handle_t)m_device); }

Module *GPUDevice::newModule(const char *moduleFile) const {
    return new gpu::Module((ze_device_handle_t)m_device, moduleFile);
}

Kernel *GPUDevice::newKernel(const Module &module, const char *name) const { return new gpu::Kernel(module, name); }

} // namespace ispcrt
