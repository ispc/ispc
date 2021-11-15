// Copyright 2020-2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "GPUDevice.h"

#if defined(_WIN32) || defined(_WIN64)

#else
#include <dlfcn.h>
#endif
// std
#include <algorithm>
#include <array>
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
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        res = ISPCRT_OUT_OF_MEMORY;
        break;
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        res = ISPCRT_DEVICE_RESET;
        break;
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        res = ISPCRT_DEVICE_LOWER_POWER;
        break;
    case ZE_RESULT_ERROR_UNINITIALIZED:
        res = ISPCRT_UNINITIALIZED;
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        res = ISPCRT_UNSUPPORTED;
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
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        res = ISPCRT_INVALID_OPERATION;
        break;
    default:
        res = ISPCRT_UNKNOWN_ERROR;
    }
    return res;
}

static const std::string getIspcrtErrorMessage(ze_result_t err) {
    switch (err) {
    case ZE_RESULT_SUCCESS:
        return "";
    case ZE_RESULT_ERROR_DEVICE_LOST:
        return "Device hung, reset, was removed, or driver update occurred.";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return "Insufficient host memory to satisfy call.";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        return "Insufficient device memory to satisfy call.";
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        return "Device requires a reset.";
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        return "Device currently in low power state.";
    case ZE_RESULT_ERROR_UNINITIALIZED:
        return "Driver is not initialized.";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        return "Unsupported version.";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        return "Unsupported feature.";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        return "Invalid argument.";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        return "Handle argument is not valid.";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        return "Pointer argument may not be nullptr.";
    case ZE_RESULT_ERROR_INVALID_SIZE:
        return "Size argument is invalid (e.g. must not be zero).";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        return "Size argument is not supported by the device (e.g. too large).";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        return "Alignment argument is not supported by the device (e.g. too small).";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        return "Enumerator argument is not valid.";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        return "Enumerator argument is not supported by the device.";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        return "Image format is not supported by the device.";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        return "Native binary is not supported by the device.";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        return "Global variable is not found in the module.";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        return "Kernel name is not found in the module.";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        return "Function name is not found in the module.";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        return "Group size dimension is not valid for the kernel or device.";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        return "Global width dimension is not valid for the kernel or device.";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        return "Kernel argument index is not valid for kernel.";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        return "Kernel argument size does not match kernel.";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        return "Value of kernel attribute is not valid for the kernel or device.";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        return "Module with imports needs to be linked before kernels can be created from it.";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        return "Command list type does not match command queue type.";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
        return "Copy operations do not support overlapping regions of memory.";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        return "Object pointed to by handle still in-use by device.";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        return "Synchronization object in invalid state.";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        return "Error occurred when building module, see build log for details.";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        return "Error occurred when linking modules, see build log for details.";
    default:
        return "Unknown or internal error.";
    }
}

#define L0_THROW_IF(status)                                                                                            \
    {                                                                                                                  \
        if (status != 0) {                                                                                             \
            std::stringstream ss;                                                                                      \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;                           \
            ss << ": " << ispcrt::gpu::getIspcrtErrorMessage(status);                                                  \
            throw ispcrt::base::ispcrt_runtime_error(ispcrt::gpu::getIspcrtError(status), ss.str());                   \
        }                                                                                                              \
    }

#define L0_SAFE_CALL(call)                                                                                             \
    { L0_THROW_IF((call)) }

#define L0_SAFE_CALL_NOEXCEPT(call)                                                                                    \
    {                                                                                                                  \
        auto status = (call);                                                                                          \
        if (status != 0) {                                                                                             \
            std::stringstream ss;                                                                                      \
            ss << __FILE__ << ":" << __LINE__ << ": L0 error 0x" << std::hex << (int)status;                           \
            ss << ": " << ispcrt::gpu::getIspcrtErrorMessage(status);                                                  \
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

    void resetEvent() { L0_SAFE_CALL(zeEventHostReset(m_handle)); }

    bool isReady() {
        ze_result_t eventStatus = zeEventQueryStatus(m_handle);
        // Event completed
        if (eventStatus == ZE_RESULT_SUCCESS) {
            return true;
        }
        return false;
    }

    bool isActive() { return m_in_use; }

    void setActive() { m_in_use = true; }

    void setFree() { m_in_use = false; }

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
    // This property tracks if event is "active" and should be used as
    // dependency for kernel launches.
    bool m_in_use{false};
};

struct CommandList {
    CommandList(ze_device_handle_t device, ze_context_handle_t context, const uint32_t ordinal)
        : m_device(device), m_context(context), m_ordinal(ordinal) {
        ze_command_list_desc_t commandListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, m_ordinal, 0};

        L0_SAFE_CALL(zeCommandListCreate(m_context, m_device, &commandListDesc, &m_handle));
        if (!m_handle)
            throw std::runtime_error("Failed to create command list!");
    }

    ze_command_list_handle_t handle() const { return m_handle; }

    ~CommandList() {
        if (m_handle)
            L0_SAFE_CALL_NOEXCEPT(zeCommandListDestroy(m_handle));
        m_events.clear();
    }

    uint32_t ordinal() { return m_ordinal; }

    void clear() {
        m_numCommands = 0;
        m_events.clear();
        m_submitted = false;
    }

    void reset() {
        if (m_numCommands > 0) {
            L0_SAFE_CALL(zeCommandListReset(m_handle));
        }
        clear();
    }

    void submit(ze_command_queue_handle_t q) {
        if (!m_submitted && m_numCommands > 0) {
            L0_SAFE_CALL(zeCommandListClose(m_handle));
            L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(q, 1, &m_handle, nullptr));
            m_submitted = true;
        }
    }

    void inc() {
        m_numCommands++;
        m_submitted = false;
    }

    uint32_t count() { return m_numCommands; }

    void addEvent(Event *event) { m_events.push_back(event); }

    std::vector<ze_event_handle_t> getEventHandlers() {
        std::vector<ze_event_handle_t> hEvents;
        for (const auto &ev : m_events) {
            hEvents.push_back(ev->handle());
        }
        return hEvents;
    }

  private:
    ze_command_list_handle_t m_handle{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_device_handle_t m_device{nullptr};
    const uint32_t m_ordinal{0};
    bool m_submitted{false};
    uint32_t m_numCommands{0};
    // List of events associated with command list
    std::vector<Event *> m_events;
};

enum class ISPCRTEventPoolType { compute, copy };

struct EventPool {
    constexpr static uint32_t POOL_SIZE_CAP = 100000;

    EventPool(ze_context_handle_t context, ze_device_handle_t device,
              ISPCRTEventPoolType type = ISPCRTEventPoolType::compute)
        : m_context(context), m_device(device) {
        // Get device timestamp resolution
        ze_device_properties_t device_properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        L0_SAFE_CALL(zeDeviceGetProperties(m_device, &device_properties));
        m_timestampFreq = device_properties.timerResolution;
        if(device_properties.kernelTimestampValidBits < 64) {
            m_timestampMaxValue = ((uint64_t)1 << device_properties.kernelTimestampValidBits) - 1;
        } else {
            // We can't calculate max using bitshifting for 64 bits
            // so simple solution for this case.
            m_timestampMaxValue = (uint64_t)-1;
        }
        // Create pool
        auto poolSize = POOL_SIZE_CAP;
        // For compute event pool check if ISPCRT_MAX_KERNEL_LAUNCHES is set
        if (type == ISPCRTEventPoolType::compute) {
            // User can set a lower limit for the pool size, which in fact limits
            // the number of possible kernel launches. To make it more clear for the user,
            // the variable is named ISPCRT_MAX_KERNEL_LAUNCHES
            constexpr const char *POOL_SIZE_ENV_NAME = "ISPCRT_MAX_KERNEL_LAUNCHES";
#if defined(_WIN32) || defined(_WIN64)
            char *poolSizeEnv = nullptr;
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
                std::cerr << "[ISPCRT][WARNING] " << POOL_SIZE_ENV_NAME << " value too large, using " << POOL_SIZE_CAP
                          << " instead." << std::endl;
            } else {
                m_poolSize = poolSize;
            }
        } else {
            m_poolSize = poolSize;
        }
        ze_event_pool_desc_t eventPoolDesc = {};
        eventPoolDesc.count = m_poolSize;
        eventPoolDesc.flags =
            (ze_event_pool_flag_t)(ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
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
        for (const auto &p : m_events_pool) {
            deleteEvent(p);
        }
        m_events_pool.clear();
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

    Event *getEvent() {
        // Get event from pool or create a new one if there is no ready event yet
        Event *event = nullptr;
        if (m_events_pool.size() > 0) {
            for (const auto &ev : m_events_pool) {
                // If event is completed, reuse it
                if (ev->isReady()) {
                    event = ev;
                    event->resetEvent();
                    event->setActive();
                    break;
                }
            }
        }
        if (event == nullptr) {
            event = createEvent();
            if (event == nullptr)
                throw std::runtime_error("Failed to create event");
            event->setActive();
            m_events_pool.push_back(event);
        }
        return event;
    }

    void releaseEvents() {
        for (const auto &ev : m_events_pool) {
            ev->setFree();
        }
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
    // Events pool for reuse
    std::vector<Event *> m_events_pool;
};

struct MemoryView : public ispcrt::base::MemoryView {
    MemoryView(ze_context_handle_t context, ze_device_handle_t device, void *appMem, size_t numBytes, bool shared)
        : m_hostPtr(appMem), m_size(numBytes), m_context(context), m_device(device), m_shared(shared) {}

    ~MemoryView() {
        if (m_devicePtr)
            L0_SAFE_CALL_NOEXCEPT(zeMemFree(m_context, m_devicePtr));
    }

    bool isShared() { return m_shared; }

    void *hostPtr() { return m_shared ? devicePtr() : m_hostPtr; };

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
            status =
                zeMemAllocShared(m_context, &device_alloc_desc, &host_alloc_desc, m_size, 64, m_device, &m_devicePtr);
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
    Module(ze_device_handle_t device, ze_context_handle_t context, const char *moduleFile, const bool is_mock_dev,
           const ISPCRTModuleOptions &opts)
        : m_file(moduleFile) {
        std::ifstream is;
        ze_module_format_t moduleFormat = ZE_MODULE_FORMAT_IL_SPIRV;
        // Try to open spv file by default if ISPCRT_USE_ZEBIN is not set.
        // TODO: change default to zebin when it gets more mature
#if defined(_WIN32) || defined(_WIN64)
        char *userZEBinFormatEnv = nullptr;
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
        // If stackSize has default value 0, do not set -stateless-stack-mem-size,
        // it will be set to 8192 in VC backend by default.
        if (opts.stackSize > 0) {
            igcOptions += " -stateless-stack-mem-size=" + std::to_string(opts.stackSize);
        }
        constexpr auto MAX_ISPCRT_IGC_OPTIONS = 2000UL;
#if defined(_WIN32) || defined(_WIN64)
        char *userIgcOptionsEnv = nullptr;
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

        // Set device/shared indirect flags
        ze_kernel_indirect_access_flags_t kernel_flags =
            ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE | ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;

        L0_SAFE_CALL(zeKernelSetIndirectAccess(m_kernel, kernel_flags));

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
    TaskQueue(ze_device_handle_t device, ze_context_handle_t context, const bool is_mock_dev)
        : m_ep_compute(context, device, ISPCRTEventPoolType::compute),
          m_ep_copy(context, device, ISPCRTEventPoolType::copy) {
        m_context = context;
        m_device = device;

        uint32_t copyOrdinal = std::numeric_limits<uint32_t>::max();
        uint32_t computeOrdinal = 0;
        // Check env variable before queue configuration
        bool isCopyEngineEnabled = true;
#if defined(_WIN32) || defined(_WIN64)
        char *is_disable_copy_eng_s = nullptr;
        size_t is_disable_copy_eng_sz = 0;
        _dupenv_s(&is_disable_copy_eng_s, &is_disable_copy_eng_sz, "ISPCRT_DISABLE_COPY_ENGINE");
        isCopyEngineEnabled = (is_disable_copy_eng_s == nullptr);
#else
        isCopyEngineEnabled = getenv("ISPCRT_DISABLE_COPY_ENGINE") == nullptr;
#endif

        bool useMultipleCommandLists = true;
#if defined(_WIN32) || defined(_WIN64)
        char *use_use_multi_cmdl_s = nullptr;
        size_t use_use_multi_cmdl_sz = 0;
        _dupenv_s(&use_use_multi_cmdl_s, &use_use_multi_cmdl_sz, "ISPCRT_DISABLE_MULTI_COMMAND_LISTS");
        useMultipleCommandLists = (use_use_multi_cmdl_s == nullptr);
#else
        useMultipleCommandLists = getenv("ISPCRT_DISABLE_MULTI_COMMAND_LISTS") == nullptr;
#endif
        // No need to create copy queue if only one command list is requested.
        if (!is_mock_dev && isCopyEngineEnabled && useMultipleCommandLists) {
            // Discover all command queue groups
            uint32_t queueGroupCount = 0;
            L0_SAFE_CALL(zeDeviceGetCommandQueueGroupProperties(device, &queueGroupCount, nullptr));
            ze_command_queue_group_properties_t *queueGroupProperties = (ze_command_queue_group_properties_t *)malloc(
                queueGroupCount * sizeof(ze_command_queue_group_properties_t));
            zeDeviceGetCommandQueueGroupProperties(device, &queueGroupCount, queueGroupProperties);

            if (queueGroupProperties != NULL) {
                for (uint32_t i = 0; i < queueGroupCount; i++) {
                    if (queueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                        computeOrdinal = i;
                        break;
                    }
                }

                for (uint32_t i = 0; i < queueGroupCount; i++) {
                    if ((queueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0 &&
                        (queueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
                        copyOrdinal = i;
                        break;
                    }
                }
            }
            free(queueGroupProperties);
        }

        if (copyOrdinal == std::numeric_limits<uint32_t>::max()) {
            copyOrdinal = computeOrdinal;
        } else {
            useCopyEngine = true;
        }

        m_cl_compute = createCommandList(computeOrdinal);
        if (!is_mock_dev && useMultipleCommandLists) {
            m_cl_mem_d2h = createCommandList(copyOrdinal);
            m_cl_mem_h2d = createCommandList(copyOrdinal);
        } else {
            m_cl_mem_d2h = m_cl_compute;
            m_cl_mem_h2d = m_cl_compute;
        }

        createCommandQueue(&m_q_compute, computeOrdinal);
        // If there is no copy engine in HW, no need to create separate queue
        if (useCopyEngine) {
            createCommandQueue(&m_q_copy, copyOrdinal);
        } else {
            m_q_copy = m_q_compute;
        }
    }

    ~TaskQueue() {
        if (m_q_compute)
            L0_SAFE_CALL_NOEXCEPT(zeCommandQueueDestroy(m_q_compute));
        if (m_q_copy && m_q_copy != m_q_compute)
            L0_SAFE_CALL_NOEXCEPT(zeCommandQueueDestroy(m_q_copy));

        // Clean up any events that could be in the queue
        for (const auto &p : m_events_compute_list) {
            auto e = p.first;
            auto f = p.second;
            // Any commands associated with this future will never
            // be executed so we mark the future as not valid
            f->m_valid = false;
            f->refDec();
            m_ep_compute.deleteEvent(e);
        }

        m_events_compute_list.clear();
    }

    void barrier() override { L0_SAFE_CALL(zeCommandListAppendBarrier(m_cl_compute->handle(), nullptr, 0, nullptr)); }

    void copyToHost(ispcrt::base::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        // Form a vector of compute events which should complete before copying memory to host
        std::vector<ze_event_handle_t> waitEvents;
        for (const auto &ev : m_events_compute_list) {
            waitEvents.push_back(ev.first->handle());
        }
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl_mem_d2h->handle(), view.hostPtr(), view.devicePtr(),
                                                   view.numBytes(), nullptr, waitEvents.size(), waitEvents.data()));

        m_cl_mem_d2h->inc();
    }

    void copyToDevice(ispcrt::base::MemoryView &mv) override {
        auto &view = (gpu::MemoryView &)mv;
        // Create event which will signal when memory copy is completed
        Event *copyEvent = m_ep_copy.getEvent();
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_cl_mem_h2d->handle(), view.devicePtr(), view.hostPtr(),
                                                   view.numBytes(), copyEvent->handle(), 0, nullptr));
        m_cl_mem_h2d->inc();
        m_cl_mem_h2d->addEvent(copyEvent);
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

        std::array<uint32_t, 3> suggestedGroupSize = {0};
        L0_SAFE_CALL(zeKernelSuggestGroupSize(kernel.handle(), dim0, dim1, dim2, &suggestedGroupSize[0],
                                              &suggestedGroupSize[1], &suggestedGroupSize[2]));
        // TODO: Is this needed? Didn't find info in spec on the valid values that zeKernelSuggestGroupSize will return
        suggestedGroupSize[0] = std::max(suggestedGroupSize[0], uint32_t(1));
        suggestedGroupSize[1] = std::max(suggestedGroupSize[1], uint32_t(1));
        suggestedGroupSize[2] = std::max(suggestedGroupSize[2], uint32_t(1));

        L0_SAFE_CALL(
            zeKernelSetGroupSize(kernel.handle(), suggestedGroupSize[0], suggestedGroupSize[1], suggestedGroupSize[2]));

        const ze_group_count_t dispatchTraits = {uint32_t(dim0) / suggestedGroupSize[0],
                                                 uint32_t(dim1) / suggestedGroupSize[1],
                                                 uint32_t(dim2) / suggestedGroupSize[2]};
        auto event = m_ep_compute.createEvent();
        if (event == nullptr)
            throw std::runtime_error("Failed to create event!");
        try {
            L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_cl_compute->handle(), kernel.handle(), &dispatchTraits,
                                                         event->handle(), m_cl_mem_h2d->getEventHandlers().size(),
                                                         m_cl_mem_h2d->getEventHandlers().data()));
            m_cl_compute->inc();
        } catch (ispcrt::base::ispcrt_runtime_error &e) {
            // cleanup and rethrow
            m_ep_compute.deleteEvent(event);
            throw e;
        }

        auto *future = new gpu::Future;
        assert(future);
        m_events_compute_list.push_back(std::make_pair(event, future));

        return future;
    }

    void sync() override {
        // Submit command lists
        submit();

        // Synchronize
        if (useCopyEngine) {
            // If there are commands to copy from device to host,
            // run sync of copy queue - it will ensure that all commands in pipeline were executed before.
            if (anyD2HCopyCommand()) {
                L0_SAFE_CALL(zeCommandQueueSynchronize(m_q_copy, std::numeric_limits<uint64_t>::max()));
            } else {
                // If there are commands in compute list, run sync of compute queue -
                // it will ensure that dependent copy commands from host to device were executed before.
                if (anyComputeCommand()) {
                    L0_SAFE_CALL(zeCommandQueueSynchronize(m_q_compute, std::numeric_limits<uint64_t>::max()));
                }
                // If there are commands in copy to device commandlist only, run sync of copy queue.
                else if (anyH2DCopyCommand()) {
                    L0_SAFE_CALL(zeCommandQueueSynchronize(m_q_copy, std::numeric_limits<uint64_t>::max()));
                }
            }
        } else {
            // If we have any command in one of our command lists, make queue sync
            if (anyD2HCopyCommand() || anyH2DCopyCommand() || anyComputeCommand()) {
                L0_SAFE_CALL(zeCommandQueueSynchronize(m_q_compute, std::numeric_limits<uint64_t>::max()));
            }
        }
        m_cl_compute->reset();
        m_cl_mem_h2d->reset();
        m_cl_mem_d2h->reset();

        // Update future objects corresponding to the events that have just completed
        for (const auto &p : m_events_compute_list) {
            auto e = p.first;
            auto f = p.second;
            ze_kernel_timestamp_result_t tsResult;
            L0_SAFE_CALL(zeEventQueryKernelTimestamp(e->handle(), &tsResult));
            if (tsResult.context.kernelEnd >= tsResult.context.kernelStart) {
                f->m_time = (tsResult.context.kernelEnd - tsResult.context.kernelStart);
            } else {
                // If we overflow kernelEnd counter then this method
                // should be used for calculate time.
                f->m_time = ((m_ep_compute.getTimestampMaxValue() - tsResult.context.kernelStart) +
                             tsResult.context.kernelEnd + 1);
            }
            f->m_time *= m_ep_compute.getTimestampRes();
            f->m_valid = true;
            f->refDec();
            m_ep_compute.deleteEvent(e);
        }

        m_events_compute_list.clear();
        m_ep_copy.releaseEvents();
    }

    void *taskQueueNativeHandle() const override { return m_q_compute; }

  private:
    ze_command_queue_handle_t m_q_compute{nullptr};
    ze_command_queue_handle_t m_q_copy{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_device_handle_t m_device{nullptr};

    CommandList *m_cl_compute{nullptr};
    CommandList *m_cl_mem_h2d{nullptr};
    CommandList *m_cl_mem_d2h{nullptr};
    EventPool m_ep_compute, m_ep_copy;
    std::vector<std::pair<Event *, Future *>> m_events_compute_list;

    bool useCopyEngine{false};

    CommandList *createCommandList(uint32_t ordinal) {
        auto cmdl = new CommandList(m_device, m_context, ordinal);
        assert(cmdl);
        return cmdl;
    }

    void createCommandQueue(ze_command_queue_handle_t *q, uint32_t ordinal) {
        // Create compute command queue
        ze_command_queue_desc_t commandQueueDesc = {};
        commandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        commandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        commandQueueDesc.ordinal = ordinal;

        L0_SAFE_CALL(zeCommandQueueCreate(m_context, m_device, &commandQueueDesc, q));

        if (q == nullptr)
            throw std::runtime_error("Failed to create command queue!");
    }

    void submit() override {
        m_cl_mem_h2d->submit(m_q_copy);
        m_cl_compute->submit(m_q_compute);
        m_cl_mem_d2h->submit(m_q_copy);
    }

    bool anyH2DCopyCommand() { return m_cl_mem_h2d->count() > 0; }
    bool anyD2HCopyCommand() { return m_cl_mem_d2h->count() > 0; }
    bool anyComputeCommand() { return m_events_compute_list.size(); }
};

// limitation: we assume that there'll be only one driver
// for Intel devices
static std::vector<ze_device_handle_t> g_deviceList;

static ze_driver_handle_t deviceDiscovery(bool *p_is_mock) {
    static ze_driver_handle_t selectedDriver = nullptr;
    bool is_mock = false;
#if defined(_WIN32) || defined(_WIN64)
    char *is_mock_s = nullptr;
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
    char *gpuDeviceEnv = nullptr;
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
    return new gpu::TaskQueue((ze_device_handle_t)m_device, (ze_context_handle_t)m_context, m_is_mock);
}

base::Module *GPUDevice::newModule(const char *moduleFile, const ISPCRTModuleOptions &opts) const {
    return new gpu::Module((ze_device_handle_t)m_device, (ze_context_handle_t)m_context, moduleFile, m_is_mock, opts);
}

base::Kernel *GPUDevice::newKernel(const base::Module &module, const char *name) const {
    return new gpu::Kernel(module, name);
}

void *GPUDevice::platformNativeHandle() const { return m_driver; }

void *GPUDevice::deviceNativeHandle() const { return m_device; }

void *GPUDevice::contextNativeHandle() const { return m_context; }

} // namespace ispcrt
