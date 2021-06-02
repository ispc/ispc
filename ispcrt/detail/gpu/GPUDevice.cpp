// Copyright 2020-2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "GPUDevice.h"

#if defined(_WIN32) || defined(_WIN64)

#else
#include <dlfcn.h>
#endif
// std
#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// ispcrt
#include "detail/Exception.h"

// level0
#include <level_zero/ze_api.h>

namespace ispcrt {
namespace gpu {

static const ISPCRTError getIspcrtError(ze_result_t err) {
    auto res = ISPCRT_UNKNOWN_ERROR;
    switch (err) {
        case ZE_RESULT_SUCCESS:
            res = ISPCRT_NO_ERROR;
            break;
        case ZE_RESULT_ERROR_DEVICE_LOST:
            res = ISPCRT_DEVICE_LOST;
            break;
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        case ZE_RESULT_ERROR_INVALID_SIZE:
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
            res = ISPCRT_INVALID_ARGUMENT;
            break;
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
            res = ISPCRT_INVALID_OPERATION;
            break;
        default:
            res = ISPCRT_UNKNOWN_ERROR;
    }
    return res;
}

#define L0_THROW_IF(status)                                                                                            \
    {                                                                                                                  \
        if (status != 0) {                                                                                             \
            std::stringstream ss;                                                                                      \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;                           \
            throw ispcrt::base::ispcrt_runtime_error(ispcrt::gpu::getIspcrtError(status), ss.str());                   \
        }                                                                                                              \
    }

#define L0_SAFE_CALL(call)                                                                                             \
    {                                                                                                                  \
        L0_THROW_IF((call))                                                                                            \
    }

#define L0_SAFE_CALL_NOEXCEPT(call)                                                                                    \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            std::stringstream ss;                                                                                      \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;                           \
            std::cerr << ss.str() << std::endl;                                                                        \
        }                                                                                                              \
    }

struct Future : public ispcrt::base::Future {
    Future() {}
    virtual ~Future() {}

    bool valid() override { return m_valid; }
    uint64_t time() override { return m_time; }

    friend class TaskQueue;

  private:
    uint64_t m_time{0};
    bool m_valid{false};
};

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
        ze_event_desc_t eventDesc = {};

        eventDesc.index = m_index;
        eventDesc.pNext = nullptr;
        eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        L0_SAFE_CALL(zeEventCreate(m_pool, &eventDesc, &m_handle));
        if (!m_handle)
            throw std::runtime_error("Failed to create event!");
    }
    ze_event_handle_t m_handle{nullptr};
    ze_event_pool_handle_t m_pool{nullptr};
    uint32_t m_index{0};
};

struct EventPool {
    constexpr static uint32_t POOL_SIZE_CAP = 100000;

    EventPool(ze_context_handle_t context, ze_device_handle_t device) : m_context(context), m_device(device) {
        // Get device timestamp resolution
        ze_device_properties_t device_properties;
        L0_SAFE_CALL(zeDeviceGetProperties(m_device, &device_properties));
        m_timestampFreq = device_properties.timerResolution;
        m_timestampMaxValue = ~(-1 << device_properties.kernelTimestampValidBits);
        // Create pool

        // User can set a lower limit for the pool size, which in fact limits
        // the number of possible kernel launches. To make it more clear for the user,
        // the variable is named ISPCRT_MAX_KERNEL_LAUNCHES
        constexpr const char* POOL_SIZE_ENV_NAME = "ISPCRT_MAX_KERNEL_LAUNCHES";
        auto poolSize = POOL_SIZE_CAP;
    #if defined(_WIN32) || defined(_WIN64)
        char* poolSizeEnv = nullptr;
        size_t poolSizeEnvSz = 0;
        _dupenv_s(&poolSizeEnv, &poolSizeEnvSz, POOL_SIZE_ENV_NAME);
    #else
        const char *poolSizeEnv = getenv(POOL_SIZE_ENV_NAME);
    #endif
        if (poolSizeEnv) {
            std::istringstream(poolSizeEnv) >> poolSize;
        }
        if (poolSize > POOL_SIZE_CAP) {
            m_poolSize = POOL_SIZE_CAP;
            std::cerr << "[ISPCRT][WARNING] " << POOL_SIZE_ENV_NAME << " value too large, using " << POOL_SIZE_CAP << " instead." << std::endl;
        } else {
            m_poolSize = poolSize;
        }

        ze_event_pool_desc_t eventPoolDesc = {};
        eventPoolDesc.count = m_poolSize;
        eventPoolDesc.flags = (ze_event_pool_flag_t)(ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
        L0_SAFE_CALL(zeEventPoolCreate(m_context, &eventPoolDesc, 1, &m_device, &m_pool));
        if (!m_pool) {
            std::stringstream ss;
            ss << "Failed to create event pool for device 0x" << std::hex << m_device << " (context 0x" << m_context
               << ")";
            throw std::runtime_error(ss.str());
        }
        // Put all event ids into a freelist
        for (uint32_t i = 0; i < m_poolSize; i++) {
            m_freeList.push_back(i);
        }
    }

    ~EventPool() {
        if (m_pool) {
            L0_SAFE_CALL_NOEXCEPT(zeEventPoolDestroy(m_pool));
        }
        assert(m_freeList.size() == m_poolSize);
        m_freeList.clear();
    }

    Event *createEvent() {
        if (m_freeList.empty()) {
            return nullptr;
        }
        auto e = new Event(m_pool, m_freeList.front());
        assert(e);
        m_freeList.pop_front();
        return e;
    }

    void deleteEvent(Event *e) {
        assert(e);
        m_freeList.push_back(e->index());
        delete e;
    }

    uint64_t getTimestampRes() const { return m_timestampFreq; }
    uint64_t getTimestampMaxValue() const { return m_timestampMaxValue; }

  private:
    ze_context_handle_t m_context{nullptr};
    ze_device_handle_t m_device{nullptr};
    ze_event_pool_handle_t m_pool{nullptr};
    uint64_t m_timestampFreq;
    uint64_t m_timestampMaxValue;
    uint32_t m_poolSize;
    std::deque<uint32_t> m_freeList;
};

struct MemoryView : public ispcrt::base::MemoryView {
    MemoryView(ze_context_handle_t context, ze_device_handle_t device, void *appMem, size_t numBytes, bool shared)
        : m_hostPtr(appMem), m_size(numBytes), m_context(context), m_device(device), m_shared(shared) {}

    ~MemoryView() {
        if (m_devicePtr)
            L0_SAFE_CALL_NOEXCEPT(zeMemFree(m_context, m_devicePtr));
    }

    bool isShared() { return m_shared; }

    void *hostPtr() {
        return m_shared ? devicePtr() : m_hostPtr;
    };

    void *devicePtr() {
        if (!m_devicePtr)
            allocate();
        return m_devicePtr;
    };

    size_t numBytes() { return m_size; };

  private:
    void allocate() {
        ze_result_t status;
        if (m_shared) {
            ze_device_mem_alloc_desc_t device_alloc_desc = {};
            ze_host_mem_alloc_desc_t host_alloc_desc = {};
            status = zeMemAllocShared(m_context, &device_alloc_desc, &host_alloc_desc,
                                      m_size, 64, m_device, &m_devicePtr);
        } else {
            ze_device_mem_alloc_desc_t allocDesc = {};
            status = zeMemAllocDevice(m_context, &allocDesc, m_size, m_size, m_device, &m_devicePtr);
        }
        if (status != ZE_RESULT_SUCCESS)
            m_devicePtr = nullptr;
        L0_THROW_IF(status);
    }

    bool m_shared{false};
    void *m_hostPtr{nullptr};
    void *m_devicePtr{nullptr};
    size_t m_size{0};

    ze_device_handle_t m_device{nullptr};
    ze_context_handle_t m_context{nullptr};
};


struct Module : public ispcrt::base::Module {
    Module(ze_device_handle_t device, ze_context_handle_t context, const char *moduleFile, bool is_mock_dev)
                : m_file(moduleFile) {
        std::ifstream is;
        ze_module_format_t moduleFormat = ZE_MODULE_FORMAT_IL_SPIRV;
        // Try to open spv file by default if ISPCRT_USE_ZEBIN is not set.
        // TODO: change default to zebin when it gets more mature
#if defined(_WIN32) || defined(_WIN64)
        char* userZEBinFormatEnv = nullptr;
        size_t userZEBinFormatEnvSz = 0;
        _dupenv_s(&userZEBinFormatEnv, &userZEBinFormatEnvSz, "ISPCRT_USE_ZEBIN");
#else
        const char *userZEBinFormatEnv = getenv("ISPCRT_USE_ZEBIN");
#endif

        size_t codeSize = 0;
        if (!is_mock_dev) {
            if (userZEBinFormatEnv) {
                is.open(m_file + ".bin", std::ios::binary);
                moduleFormat = ZE_MODULE_FORMAT_NATIVE;
                if (!is.good())
                    throw std::runtime_error("Failed to open zebin file!");
            } else {
                // Try to read .spv file by default
                is.open(m_file + ".spv", std::ios::binary);
                if (!is.good())
                    throw std::runtime_error("Failed to open spv file!");
            }

            is.seekg(0, std::ios::end);
            codeSize = is.tellg();
            is.seekg(0, std::ios::beg);

            if (codeSize == 0)
                throw std::runtime_error("Code size is '0'!");

            m_code.resize(codeSize);

            is.read((char *)m_code.data(), codeSize);
            is.close();
        }

        // Collect potential additional options for the compiler from the environment.
        // We assume some default options for the compiler, but we also
        // allow adding more options by the user. The content of the
        // ISPCRT_IGC_OPTIONS variable should be prefixed by the user with
        // + or = sign. '+' means that the content of the variable should
        // be added to the default igc options, while '=' will replace
        // the options with the content of the env var.
        std::string igcOptions = "-vc-codegen -no-optimize -Xfinalizer '-presched'";
        constexpr auto MAX_ISPCRT_IGC_OPTIONS = 2000UL;
#if defined(_WIN32) || defined(_WIN64)
        char* userIgcOptionsEnv = nullptr;
        size_t userIgcOptionsEnvSz = 0;
        _dupenv_s(&userIgcOptionsEnv, &userIgcOptionsEnvSz, "ISPCRT_IGC_OPTIONS");
#else
        const char *userIgcOptionsEnv = getenv("ISPCRT_IGC_OPTIONS");
#endif
        if (userIgcOptionsEnv) {
            // Copy at most MAX_ISPCRT_IGC_OPTIONS characters from the env - just to be safe
            const auto copyChars = std::min(std::strlen(userIgcOptionsEnv), (size_t)MAX_ISPCRT_IGC_OPTIONS);
            std::string userIgcOptions(userIgcOptionsEnv, copyChars);
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

        ze_module_desc_t moduleDesc = {};
        moduleDesc.format = moduleFormat;
        moduleDesc.inputSize = codeSize;
        moduleDesc.pInputModule = m_code.data();
        moduleDesc.pBuildFlags = igcOptions.c_str();

        assert(device != nullptr);
        L0_SAFE_CALL(zeModuleCreate(context, device, &moduleDesc, &m_module, nullptr));
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

struct Kernel : public ispcrt::base::Kernel {
    Kernel(const ispcrt::base::Module &_module, const char *name) : m_fcnName(name), m_module(&_module) {
        const gpu::Module &module = (const gpu::Module &)_module;

        ze_kernel_desc_t kernelDesc = {};
        kernelDesc.pKernelName = name;
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

    const ispcrt::base::Module *m_module{nullptr};
    ze_kernel_handle_t m_kernel{nullptr};
};

struct TaskQueue : public ispcrt::base::TaskQueue {
    TaskQueue(ze_device_handle_t device, ze_context_handle_t context) : m_ep(context, device) {
        ze_command_list_desc_t commandListDesc = {};
        L0_SAFE_CALL(zeCommandListCreate(context, device, &commandListDesc, &m_cl));
        if (m_cl == nullptr)
            throw std::runtime_error("Failed to create command list!");
        // Create a command queue
        ze_command_queue_desc_t commandQueueDesc = {};
        commandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        commandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        L0_SAFE_CALL(zeCommandQueueCreate(context, device, &commandQueueDesc, &m_q));
        if (m_q == nullptr)
            throw std::runtime_error("Failed to create command queue!");
    }

    ~TaskQueue() {
        if (m_q)
            L0_SAFE_CALL_NOEXCEPT(zeCommandQueueDestroy(m_q));
        if (m_cl)
            L0_SAFE_CALL_NOEXCEPT(zeCommandListDestroy(m_cl));
        // Clean up any events that could be in the queue
        for (const auto &p : m_events) {
            auto e = p.first;
            auto f = p.second;
            // Any commands associated with this future will never
            // be executed so we mark the future as not valid
            f->m_valid = false;
            f->refDec();
            m_ep.deleteEvent(e);
        }
        m_events.clear();
    }

    void barrier() override { L0_SAFE_CALL(zeCommandListAppendBarrier(m_cl, nullptr, 0, nullptr)); }

    void copyToHost(ispcrt::base::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl, view.hostPtr(), view.devicePtr(), view.numBytes(), nullptr, 0,
                                                   nullptr));
    }

    void copyToDevice(ispcrt::base::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl, view.devicePtr(), view.hostPtr(), view.numBytes(), nullptr, 0,
                                                   nullptr));
    }

    ispcrt::base::Future *launch(ispcrt::base::Kernel &k, ispcrt::base::MemoryView *params, size_t dim0, size_t dim1,
                                 size_t dim2) override {
        auto &kernel = (gpu::Kernel &)k;

        void *param_ptr = nullptr;
        if (params)
            param_ptr = params->devicePtr();

        // If param_ptr is nullptr, it was not set on host, so do not set kernel argument.
        if (param_ptr != nullptr) {
            L0_SAFE_CALL(zeKernelSetArgumentValue(kernel.handle(), 0, sizeof(void *), &param_ptr));
        }
        // Set indirect flag to allow USM access
        ze_kernel_indirect_access_flags_t kernel_flags = ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
        L0_SAFE_CALL(zeKernelSetIndirectAccess(kernel.handle(), kernel_flags));
        ze_group_count_t dispatchTraits = {uint32_t(dim0), uint32_t(dim1), uint32_t(dim2)};
        auto event = m_ep.createEvent();
        if (event == nullptr)
            throw std::runtime_error("Failed to create event!");
        try {
            L0_SAFE_CALL(
                zeCommandListAppendLaunchKernel(m_cl, kernel.handle(), &dispatchTraits, event->handle(), 0, nullptr));
        } catch (ispcrt::base::ispcrt_runtime_error &e) {
            // cleanup and rethrow
            m_ep.deleteEvent(event);
            throw e;
        }

        auto *future = new gpu::Future;
        assert(future);
        m_events.push_back(std::make_pair(event, future));

        return future;
    }

    void submit() override {
        if (m_submitted)
            return;
        L0_SAFE_CALL(zeCommandListClose(m_cl));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_q, 1, &m_cl, nullptr));
        m_submitted = true;
    }

    void sync() override {
        if (!m_submitted)
            submit();
        L0_SAFE_CALL(zeCommandQueueSynchronize(m_q, std::numeric_limits<uint64_t>::max()));
        L0_SAFE_CALL(zeCommandListReset(m_cl));
        // Update future objects corresponding to the events that have just completed
        for (const auto &p : m_events) {
            auto e = p.first;
            auto f = p.second;
            ze_kernel_timestamp_result_t tsResult;
            L0_SAFE_CALL(zeEventQueryKernelTimestamp(e->handle(), &tsResult));
            if (tsResult.context.kernelEnd >= tsResult.context.kernelStart) {
                f->m_time = (tsResult.context.kernelEnd - tsResult.context.kernelStart);
            } else {
                f->m_time =
                    ((m_ep.getTimestampMaxValue() - tsResult.context.kernelStart) + tsResult.context.kernelEnd + 1);
            }
            f->m_time *= m_ep.getTimestampRes();
            f->m_valid = true;
            f->refDec();
            m_ep.deleteEvent(e);
        }
        m_events.clear();
        m_submitted = false;
    }

    void* taskQueueNativeHandle() const override {
        return m_q;
    }

  private:
    ze_command_queue_handle_t m_q{nullptr};
    ze_command_list_handle_t m_cl{nullptr};
    bool  m_submitted{false};
    EventPool m_ep;
    std::vector<std::pair<Event *, Future *>> m_events;
};


// limitation: we assume that there'll be only one driver
// for Intel devices
static std::vector<ze_device_handle_t> g_deviceList;

static ze_driver_handle_t deviceDiscovery(bool *p_is_mock) {
    static ze_driver_handle_t selectedDriver = nullptr;
    bool is_mock = false;
#if defined(_WIN32) || defined(_WIN64)
    char* is_mock_s = nullptr;
    size_t is_mock_sz = 0;
    _dupenv_s(&is_mock_s, &is_mock_sz, "ISPCRT_MOCK_DEVICE");
    is_mock = is_mock_s != nullptr;
#else
    is_mock = getenv("ISPCRT_MOCK_DEVICE") != nullptr;
#endif

    // Allow reinitialization of device list for mock device
    if (!is_mock && selectedDriver != nullptr)
        return selectedDriver;

    g_deviceList.clear();

    // zeInit can be called multiple times
    L0_SAFE_CALL(zeInit(0));

    // Discover all the driver instances
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));

    if (driverCount == 0)
        throw std::runtime_error("could not find L0 driver");

    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers.data()));

    // Find only instances of Intel GPU device
    // But we can use a mock device driver for testing
    for (auto &driver : allDrivers) {
        uint32_t deviceCount = 0;
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, nullptr));
        std::vector<ze_device_handle_t> allDevices(deviceCount);
        L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, allDevices.data()));
        for (auto &device : allDevices) {
            ze_device_properties_t device_properties;
            L0_SAFE_CALL(zeDeviceGetProperties(device, &device_properties));
            if (device_properties.type == ZE_DEVICE_TYPE_GPU && device_properties.vendorId == 0x8086) {
                if (selectedDriver != nullptr && driver != selectedDriver)
                    throw std::runtime_error("there should be only one Intel driver in the system");
                selectedDriver = driver;
                g_deviceList.push_back(device);
            }
        }
    }
    if (p_is_mock != nullptr)
        *p_is_mock = is_mock;
    return selectedDriver;
}

uint32_t deviceCount() {
    deviceDiscovery(nullptr);
    return g_deviceList.size();
}

ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx) {
    deviceDiscovery(nullptr);
    if (deviceIdx >= g_deviceList.size())
        throw std::runtime_error("Invalid device number");
    ISPCRTDeviceInfo info;
    ze_device_properties_t dp;
    L0_SAFE_CALL(zeDeviceGetProperties(g_deviceList[deviceIdx], &dp));
    info.deviceId = dp.deviceId;
    info.vendorId = dp.vendorId;
    return info;
}

} // namespace gpu


// Use the first available device by default for now.
// Later we may do something more sophisticated (e.g. use the one
// with most FLOPs or have some kind of load balancing)
GPUDevice::GPUDevice() : GPUDevice(0) {}

GPUDevice::GPUDevice(uint32_t deviceIdx) {
    // Find an instance of Intel GPU device
    // User can select particular device using env variable
    // By default first available device is selected
    auto gpuDeviceToGrab = deviceIdx;
#if defined(_WIN32) || defined(_WIN64)
    char* gpuDeviceEnv = nullptr;
    size_t gpuDeviceEnvSz = 0;
    _dupenv_s(&gpuDeviceEnv, &gpuDeviceEnvSz, "ISPCRT_GPU_DEVICE");
#else
    const char *gpuDeviceEnv = getenv("ISPCRT_GPU_DEVICE");
#endif
    if (gpuDeviceEnv) {
        std::istringstream(gpuDeviceEnv) >> gpuDeviceToGrab;
    }

    // Perform GPU discovery
    m_driver = gpu::deviceDiscovery(&m_is_mock);

    if (gpuDeviceToGrab >= gpu::g_deviceList.size())
        throw std::runtime_error("could not find a valid GPU device");

    m_device = gpu::g_deviceList[gpuDeviceToGrab];

    ze_context_desc_t contextDesc = {}; // use default values
    L0_SAFE_CALL(zeContextCreate((ze_driver_handle_t)m_driver, &contextDesc, (ze_context_handle_t *)&m_context));

    if (!m_context)
        throw std::runtime_error("failed to create GPU context");
}

GPUDevice::~GPUDevice() {
    if (m_context)
        L0_SAFE_CALL_NOEXCEPT(zeContextDestroy((ze_context_handle_t)m_context));
}

base::MemoryView *GPUDevice::newMemoryView(void *appMem, size_t numBytes, bool shared) const {
    return new gpu::MemoryView((ze_context_handle_t)m_context, (ze_device_handle_t)m_device, appMem, numBytes, shared);
}

base::TaskQueue *GPUDevice::newTaskQueue() const {
    return new gpu::TaskQueue((ze_device_handle_t)m_device, (ze_context_handle_t)m_context);
}

base::Module *GPUDevice::newModule(const char *moduleFile) const {
    return new gpu::Module((ze_device_handle_t)m_device, (ze_context_handle_t)m_context, moduleFile, m_is_mock);
}

base::Kernel *GPUDevice::newKernel(const base::Module &module, const char *name) const {
    return new gpu::Kernel(module, name);
}

void *GPUDevice::platformNativeHandle() const {
    return m_driver;
}

void *GPUDevice::deviceNativeHandle() const {
    return m_device;
}

void *GPUDevice::contextNativeHandle() const {
    return m_context;
}

} // namespace ispcrt
