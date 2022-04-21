// Copyright 2020-2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "CPUDevice.h"

#if defined(_WIN32) || defined(_WIN64)
#include "windows.h"
#else
#include <dlfcn.h>
#endif
// std
#include <cassert>
#include <chrono>
#include <cstring>
#include <exception>
#include <string>

namespace ispcrt {
namespace cpu {

struct Future : public ispcrt::base::Future {
    Future() = default;
    virtual ~Future() = default;

    bool valid() override { return m_valid; }
    uint64_t time() override { return m_time; }

    friend struct TaskQueue;

  private:
    uint64_t m_time{0};
    bool m_valid{false};
};

using CPUKernelEntryPoint = void (*)(void *, size_t, size_t, size_t);

struct MemoryView : public ispcrt::base::MemoryView {
    MemoryView(void *appMem, size_t numBytes, bool shared) : m_hostPtr(appMem), m_devicePtr(appMem), m_size(numBytes), m_shared(shared) {}

    ~MemoryView() {
        if (!m_external_alloc && m_devicePtr)
            free(m_devicePtr);
    }

    bool isShared() { return m_shared; }

    void *hostPtr() {
        if (m_shared) {
            return devicePtr();
        }
        else {
            if (!m_hostPtr)
                 throw std::logic_error("pointer to the host memory is NULL");
            return m_hostPtr;
        }
    };

    void *devicePtr() {
        if (!m_devicePtr)
            allocate();
        return m_devicePtr;
    };

    size_t numBytes() { return m_size; };

  private:
    void allocate() {
        m_devicePtr = malloc(m_size);
        if (!m_devicePtr)
            throw std::bad_alloc();
        m_external_alloc = false;
    }
    bool m_external_alloc{true};
    bool m_shared{false};
    void *m_hostPtr{nullptr};
    void *m_devicePtr{nullptr};
    size_t m_size{0};
};

struct Module : public ispcrt::base::Module {
    Module(const char *moduleFile) : m_file(moduleFile) {
        if (!m_file.empty()) {
#if defined(__MACOSX__) || defined(__APPLE__)
            std::string ext = ".dylib";
#elif defined(_WIN32) || defined(_WIN64)
            std::string ext = ".dll";
#else
            std::string ext = ".so";
#endif
#if defined _WIN32
            m_lib = LoadLibrary((m_file + ext).c_str());
#else
            m_lib = dlopen(("lib" + m_file + ext).c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif

            if (!m_lib)
                throw std::logic_error("could not open CPU shared module file");
        }
    }

    ~Module() {
        if (m_lib)
#if defined(_WIN32) || defined(_WIN64)
            FreeLibrary((HMODULE)m_lib);
#else
            dlclose(m_lib);
#endif
    }

    void *lib() const { return m_lib; }

  private:
    std::string m_file;
    void *m_lib{nullptr};
};

struct Kernel : public ispcrt::base::Kernel {
    Kernel(const ispcrt::base::Module &_module, const char *_name) : m_fcnName(_name), m_module(&_module) {
        const cpu::Module &module = (const cpu::Module &)_module;

        auto name = std::string(_name) + "_cpu_entry_point";
#if defined(_WIN32) || defined(_WIN64)
        void *fcn = GetProcAddress((HMODULE)module.lib(), name.c_str());
#else
        void *fcn = dlsym(module.lib() ? module.lib() : RTLD_DEFAULT, name.c_str());
#endif

        if (!fcn)
            throw std::logic_error("could not find CPU kernel function");

        m_fcn = (CPUKernelEntryPoint)fcn;
        m_module->refInc();
    }

    ~Kernel() {
        if (m_module)
            m_module->refDec();
    }

    CPUKernelEntryPoint entryPoint() const { return m_fcn; }

  private:
    std::string m_fcnName;
    CPUKernelEntryPoint m_fcn{nullptr};

    const ispcrt::base::Module *m_module{nullptr};
};

struct TaskQueue : public ispcrt::base::TaskQueue {
    TaskQueue() {
        // no-op
    }

    void barrier() override {
        // no-op
    }

    void copyToHost(ispcrt::base::MemoryView &) override {
        // no-op
    }

    void copyToDevice(ispcrt::base::MemoryView &) override {
        // no-op
    }

    void copyMemoryView(base::MemoryView &mv_dst, base::MemoryView &mv_src, const size_t size) override {
        auto &view_dst = (cpu::MemoryView &)mv_dst;
        auto &view_src = (cpu::MemoryView &)mv_src;
        memcpy(view_dst.devicePtr(), view_src.devicePtr(), size);
    }

    ispcrt::base::Future *launch(ispcrt::base::Kernel &k, ispcrt::base::MemoryView *params, size_t dim0, size_t dim1,
                                 size_t dim2) override {
        auto &kernel = (cpu::Kernel &)k;
        auto *parameters = (cpu::MemoryView *)params;

        auto *fcn = kernel.entryPoint();

        auto *future = new cpu::Future;
        assert(future);

        auto start = std::chrono::high_resolution_clock::now();
        fcn(parameters ? parameters->devicePtr() : nullptr, dim0, dim1, dim2);
        auto end = std::chrono::high_resolution_clock::now();

        future->m_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        future->m_valid = true;

        return future;
    }

    void submit() override {
        // no-op
    }

    void sync() override {
        // no-op
    }

    void *taskQueueNativeHandle() const override { return nullptr; }
};

uint32_t deviceCount() { return 1; }

ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx) {
    ISPCRTDeviceInfo info;
    info.deviceId = 0; // for CPU we don't support it yet
    info.vendorId = 0;
    return info;
}

} // namespace cpu

ispcrt::base::MemoryView *CPUDevice::newMemoryView(void *appMem, size_t numBytes, bool shared) const {
    return new cpu::MemoryView(appMem, numBytes, shared);
}

ispcrt::base::TaskQueue *CPUDevice::newTaskQueue() const { return new cpu::TaskQueue(); }

ispcrt::base::Module *CPUDevice::newModule(const char *moduleFile, const ISPCRTModuleOptions &moduleOpts) const {
    return new cpu::Module(moduleFile);
}

ispcrt::base::Kernel *CPUDevice::newKernel(const ispcrt::base::Module &module, const char *name) const {
    return new cpu::Kernel(module, name);
}

void *CPUDevice::platformNativeHandle() const { return nullptr; }

void *CPUDevice::deviceNativeHandle() const { return nullptr; }

void *CPUDevice::contextNativeHandle() const { return nullptr; }

ISPCRTAllocationType CPUDevice::getMemAllocType(void* appMemory) const {
    return ISPCRT_ALLOC_TYPE_UNKNOWN;
}
} // namespace ispcrt
