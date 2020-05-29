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
#include <iostream>
#include <sstream>
#include <cassert>
#include <deque>
// level0
#include <level_zero/ze_api.h>

#define L0_SAFE_CALL(call)                                                                    \
    {                                                                                         \
        auto status = (call);                                                                 \
        if (status != 0) {                                                                    \
            std::stringstream ss;                                                             \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;  \
            throw std::runtime_error(ss.str());                                               \
        }                                                                                     \
    }

#define L0_SAFE_CALL_NOEXCEPT(call)                                                           \
    {                                                                                         \
        auto status = (call);                                                                 \
        if (status != 0) {                                                                    \
            std::stringstream ss;                                                             \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;  \
            std::cerr << ss.str() << std::endl;                                               \
        }                                                                                     \
    }

namespace ispcrt {
namespace gpu {

struct Event {
    Event(ze_event_pool_handle_t pool, uint32_t index) : m_pool(pool), m_index(index) {}

    ze_event_handle_t handle() {
        if (!m_handle)
            create();
        return m_handle;
    }

    ~Event() {
        if (m_handle)
            L0_SAFE_CALL_NOEXCEPT(zeEventDestroy(m_handle));
    }

    uint32_t index() { return m_index; }

  private:
    void create() {
        ze_event_desc_t eventDesc;

        eventDesc.index = m_index;
        eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        eventDesc.version = ZE_EVENT_DESC_VERSION_CURRENT;
    
        L0_SAFE_CALL(zeEventCreate(m_pool, &eventDesc, &m_handle));

        if (!m_handle)
            throw std::runtime_error("Failed to create event!");
    }
    ze_event_handle_t m_handle {nullptr};
    ze_event_pool_handle_t m_pool {nullptr};
    uint32_t m_index {0};
};

struct EventPool {
    constexpr static uint32_t POOL_SIZE = 100;

    EventPool(ze_driver_handle_t driver, ze_device_handle_t device) : m_driver(driver), m_device(device) {
        // Get device timestamp resolution
        ze_device_properties_t device_properties;
        L0_SAFE_CALL(zeDeviceGetProperties(m_device, &device_properties));
        m_timestampFreq = device_properties.timerResolution;

        // Create pool
        ze_event_pool_desc_t eventPoolDesc;
        eventPoolDesc.count = POOL_SIZE;
        // JZTODO: shouldn't it be host-visible?
        // ZE_EVENT_POOL_FLAG_HOST_VISIBLE
        eventPoolDesc.flags = (ze_event_pool_flag_t)(ZE_EVENT_POOL_FLAG_TIMESTAMP);
        eventPoolDesc.version = ZE_EVENT_POOL_DESC_VERSION_CURRENT;
        L0_SAFE_CALL(zeEventPoolCreate(m_driver, &eventPoolDesc, 1, &m_device, &m_pool));
        if (!m_pool) {
            std::stringstream ss;
            ss << "Failed to create event pool for device 0x" << std::hex << m_device << " (driver 0x" << m_driver << ")";
            throw std::runtime_error(ss.str());
        }
        // Put all event ids into a freelist
        for (uint32_t i = 0; i < POOL_SIZE; i++) {
            m_freeList.push_back(i);
        }
    }

    ~EventPool() {
        if (m_pool) {
            L0_SAFE_CALL_NOEXCEPT(zeEventPoolDestroy(m_pool));
        }
        assert(m_freeList.size() == POOL_SIZE);
        m_freeList.clear();
    }

    Event* createEvent() { 
        if (m_freeList.empty()) {
            return nullptr;
        }
        auto e = new Event(m_pool, m_freeList.front());
        assert(e);
        m_freeList.pop_front();
        return e;
    }
    
    void deleteEvent(Event* e) { 
        assert(e);
        m_freeList.push_back(e->index());
        delete e;
    }

    uint64_t getTimestampRes() const { return m_timestampFreq; }

  private:
    ze_driver_handle_t m_driver {nullptr};
    ze_device_handle_t m_device {nullptr};
    ze_event_pool_handle_t m_pool {nullptr};
    uint64_t m_timestampFreq;
    std::deque<uint32_t> m_freeList;
};

struct MemoryView : public ispcrt::MemoryView {
    MemoryView(ze_driver_handle_t driver, ze_device_handle_t device, void *appMem, size_t numBytes)
        : m_hostPtr(appMem), m_size(numBytes), m_driver(driver), m_device(device) {}

    ~MemoryView() {
        if (m_devicePtr)
            L0_SAFE_CALL_NOEXCEPT(zeDriverFreeMem(m_driver, m_devicePtr));
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
        assert(device != nullptr);
        L0_SAFE_CALL(zeModuleCreate(device, &moduleDesc, &m_module, nullptr));
        assert(m_module != nullptr);

        if (m_module == nullptr)
            throw std::runtime_error("Failed to load spv module!");
    }

    ~Module() {
        if (m_module)
            L0_SAFE_CALL_NOEXCEPT(zeModuleDestroy(m_module));
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
    TaskQueue(ze_device_handle_t device, ze_driver_handle_t driver) : m_ep(driver, device) {
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
            L0_SAFE_CALL_NOEXCEPT(zeCommandQueueDestroy(m_q));
        if (m_cl)
            L0_SAFE_CALL_NOEXCEPT(zeCommandListDestroy(m_cl));
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

    Future* launch(ispcrt::Kernel &k, ispcrt::MemoryView *params, size_t dim0, size_t dim1, size_t dim2) override {
        auto &kernel = (gpu::Kernel &)k;

        auto *future = new Future;
        assert(future);

        void *param_ptr = nullptr;

        if (params)
            param_ptr = params->devicePtr();

        L0_SAFE_CALL(zeKernelSetArgumentValue(kernel.handle(), 0, sizeof(void *), &param_ptr));

        ze_group_count_t dispatchTraits = {uint32_t(dim0), uint32_t(dim1), uint32_t(dim2)};
        auto event = m_ep.createEvent();
        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_cl, kernel.handle(), &dispatchTraits, event->handle(), 0, nullptr));
        m_events.push_back(std::make_pair(event, future));

        return future;
    }

    void sync() override {
        L0_SAFE_CALL(zeCommandListClose(m_cl));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_q, 1, &m_cl, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(m_q, std::numeric_limits<uint32_t>::max()));
        L0_SAFE_CALL(zeCommandListReset(m_cl));
        // Update future objects corresponding to the events that have just completed
        for (const auto& p : m_events) {
            auto *e = p.first; 
            auto *f = p.second;
            uint64_t contextStart, contextEnd;
            L0_SAFE_CALL(zeEventGetTimestamp(e->handle(), ZE_EVENT_TIMESTAMP_CONTEXT_START, &contextStart));
            L0_SAFE_CALL(zeEventGetTimestamp(e->handle(), ZE_EVENT_TIMESTAMP_CONTEXT_END, &contextEnd));
            f->time = (contextEnd - contextStart) * m_ep.getTimestampRes();
            f->valid = true;
            m_ep.deleteEvent(e);
        }
        m_events.clear();
    }

  private:
    ze_command_queue_handle_t m_q{nullptr};
    ze_command_list_handle_t m_cl{nullptr};
    EventPool m_ep;
    std::vector<std::pair<Event*, Future*>> m_events;
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

TaskQueue *GPUDevice::newTaskQueue() const { return new gpu::TaskQueue((ze_device_handle_t)m_device, (ze_driver_handle_t)m_driver); }

Module *GPUDevice::newModule(const char *moduleFile) const {
    return new gpu::Module((ze_device_handle_t)m_device, moduleFile);
}

Kernel *GPUDevice::newKernel(const Module &module, const char *name) const { return new gpu::Kernel(module, name); }

} // namespace ispcrt
