// Copyright 2020-2023, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "CPUDevice.h"
#include "CPUContext.h"

#if defined(_WIN32) || defined(_WIN64)
#include "windows.h"
#else
#include <dlfcn.h>
#endif
// std
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

extern "C" {
ispcrt::base::Device *load_cpu_device() { return new ispcrt::CPUDevice; }
uint32_t cpu_device_count() { return ispcrt::cpu::deviceCount(); }
ISPCRTDeviceInfo cpu_device_info(uint32_t idx) { return ispcrt::cpu::deviceInfo(idx); }
ispcrt::base::Context *load_cpu_context() { return new ispcrt::CPUContext; }
}

namespace ispcrt {
namespace cpu {

struct Future : public ispcrt::base::Future {
    Future() = default;
    virtual ~Future() = default;

    bool valid() override { return m_valid; }
    uint64_t time() override { return m_time; }

    friend struct TaskQueue;
    friend struct CommandListImpl;

  private:
    uint64_t m_time{0};
    bool m_valid{false};
};

struct Fence : public ispcrt::base::Fence {
    Fence() {}
    virtual ~Fence() {}

    void sync() override {
        // no-op
    }

    ISPCRTFenceStatus status() const override { return ISPCRT_FENCE_SIGNALED; }

    void reset() override {
        // no-op
    }

    void *nativeHandle() const override { return nullptr; }
};

using CPUKernelEntryPoint = void (*)(void *, size_t, size_t, size_t);

struct MemoryView : public ispcrt::base::MemoryView {
    MemoryView(void *appMem, size_t numBytes, bool shared)
        : m_shared(shared), m_hostPtr(appMem), m_devicePtr(appMem), m_size(numBytes) {}

    ~MemoryView() {
        if (!m_external_alloc && m_devicePtr)
            free(m_devicePtr);
    }

    bool isShared() { return m_shared; }

    void *hostPtr() {
        if (m_shared) {
            return devicePtr();
        } else {
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

struct ModuleOptions : public ispcrt::base::ModuleOptions {
    ModuleOptions() = default;
    ModuleOptions(ISPCRTModuleType moduleType, bool libraryCompilation, uint32_t stackSize)
        : m_moduleType{moduleType}, m_libraryCompilation{libraryCompilation}, m_stackSize{stackSize} {}

    uint32_t stackSize() const { return m_stackSize; }
    bool libraryCompilation() const { return m_libraryCompilation; }
    ISPCRTModuleType moduleType() const { return m_moduleType; }

    void setStackSize(uint32_t size) { m_stackSize = size; }
    void setLibraryCompilation(bool isLibraryCompilation) { m_libraryCompilation = isLibraryCompilation; }
    void setModuleType(ISPCRTModuleType type) { m_moduleType = type; }

  private:
    ISPCRTModuleType m_moduleType{ISPCRTModuleType::ISPCRT_VECTOR_MODULE};
    bool m_libraryCompilation{false};
    uint32_t m_stackSize{0};
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
            void *lib = nullptr;
#if defined _WIN32
            // Removes CWD from the search path to reduce the risk of DLL injection.
            SetDllDirectory("");
            lib = LoadLibraryEx((m_file + ext).c_str(), NULL, 0);
#else
            lib = dlopen(("lib" + m_file + ext).c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif

            if (!lib)
                throw std::logic_error("could not open CPU shared module file lib" + m_file + ext);
            m_libs.push_back(lib);
        }
    }

    Module(Module **modules, const uint32_t numModules) {
        for (uint32_t i = 0; i < numModules; i++) {
            for (auto lib : modules[i]->libs()) {
                m_libs.push_back(lib);
            }
        }
    }

    ~Module() {
        if (m_libs.size() > 0) {
            for (auto lib : m_libs) {
                if (lib) {
#if defined(_WIN32) || defined(_WIN64)
                    FreeLibrary((HMODULE)lib);
#else
                    dlclose(lib);
#endif
                }
            }
        }
    }

    void *functionPtr(const char *name) const override {
        void *fptr = nullptr;
        for (auto lib : m_libs) {
#if defined(_WIN32) || defined(_WIN64)
            fptr = GetProcAddress((HMODULE)lib, name);
#else
            fptr = dlsym(lib ? lib : RTLD_DEFAULT, name);
#endif
            if (fptr != nullptr)
                break;
        }
        if (!fptr)
            throw std::logic_error("could not find CPU function");
        return fptr;
    }

    std::vector<void *> libs() { return m_libs; };

  private:
    std::string m_file;
    std::vector<void *> m_libs;
};

struct Kernel : public ispcrt::base::Kernel {
    Kernel(const ispcrt::base::Module &_module, const char *_name) : m_fcnName(_name), m_module(&_module) {
        const cpu::Module &module = (const cpu::Module &)_module;

        auto name = std::string(_name) + "_cpu_entry_point";
        void *fcn = module.functionPtr(name.c_str());

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

struct CommandListImpl : ispcrt::base::CommandList {
    CommandListImpl() { /* no-op */
    }
    ~CommandListImpl() {
        clearFences();
        clearFutures();
    }

    void barrier() override { /* no-op */
    }

    ispcrt::base::Future *copyToHost(ispcrt::base::MemoryView &) override {
        Future *f = new Future();
        m_futures.push_back(f);
        return f;
    }

    ispcrt::base::Future *copyToDevice(ispcrt::base::MemoryView &) override {
        Future *f = new Future();
        m_futures.push_back(f);
        return f;
    }

    ispcrt::base::Future *copyMemoryView(base::MemoryView &mv_dst, base::MemoryView &mv_src,
                                         const size_t size) override {
        auto view_dst_ptr = static_cast<std::byte *>(((cpu::MemoryView &)mv_dst).devicePtr());
        auto view_src_ptr = static_cast<std::byte *>(((cpu::MemoryView &)mv_src).devicePtr());
        std::copy(view_src_ptr, view_src_ptr + size, view_dst_ptr);
        Future *f = new Future();
        m_futures.push_back(f);
        return f;
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

        if (m_timestamps) {
            future->m_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        future->m_valid = true;

        m_futures.push_back(future);
        return future;
    }

    void close() override { /* no-op */
    }

    ispcrt::base::Fence *submit() override {
        Fence *f = new Fence;
        m_fences.push_back(f);
        return f;
    }

    void reset() override {
        clearFences();
        clearFutures();
    }

    // This has no effect at the moment.
    void enableTimestamps() override { m_timestamps = true; }

    void *nativeHandle() const override { return nullptr; }

  private:
    bool m_timestamps{false};

    std::vector<Future *> m_futures;
    std::vector<Fence *> m_fences;

    void clearFences() {
        if (m_fences.size()) {
            for (const auto &f : m_fences) {
                f->refDec();
            }
            m_fences.clear();
        }
    }

    void clearFutures() {
        if (m_futures.size()) {
            for (const auto &f : m_futures) {
                f->refDec();
            }
            m_futures.clear();
        }
    }
};

struct CommandQueueImpl : ispcrt::base::CommandQueue {
    CommandQueueImpl() { /* no-op */
    }

    ~CommandQueueImpl() { clearCommandList(); }

    ispcrt::base::CommandList *createCommandList() override {
        CommandListImpl *p = new CommandListImpl();
        m_cmdlists.push_back(p);
        return p;
    }

    void sync() override { /* no-op */
    }

    void *nativeHandle() const override { return nullptr; }

  private:
    std::vector<CommandListImpl *> m_cmdlists;

    void clearCommandList() {
        if (m_cmdlists.size()) {
            for (const auto &l : m_cmdlists) {
                l->refDec();
            }
            m_cmdlists.clear();
        }
    }
};

struct TaskQueue : public ispcrt::base::TaskQueue {
    TaskQueue() {
        // no-op
    }

    ~TaskQueue() {
        for (auto f : m_futures) {
            delete f;
        }
        m_futures.clear();
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
        auto view_dst_ptr = static_cast<std::byte *>(((cpu::MemoryView &)mv_dst).devicePtr());
        auto view_src_ptr = static_cast<std::byte *>(((cpu::MemoryView &)mv_src).devicePtr());
        std::copy(view_src_ptr, view_src_ptr + size, view_dst_ptr);
    }

    ispcrt::base::Future *launch(ispcrt::base::Kernel &k, ispcrt::base::MemoryView *params, size_t dim0, size_t dim1,
                                 size_t dim2) override {
        auto &kernel = (cpu::Kernel &)k;
        auto *parameters = (cpu::MemoryView *)params;

        auto *fcn = kernel.entryPoint();

        auto *future = new cpu::Future;
        assert(future);
        // Vector to know what to deallocate when TaskQueue object destructed
        m_futures.push_back(future);

        auto start = std::chrono::high_resolution_clock::now();
        fcn(parameters ? parameters->devicePtr() : nullptr, dim0, dim1, dim2);
        auto end = std::chrono::high_resolution_clock::now();

        future->m_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        future->m_valid = true;

        return future;
    }

    void sync() override {
        // no-op
    }

    void *taskQueueNativeHandle() const override { return nullptr; }

  private:
    std::vector<cpu::Future *> m_futures;
};

uint32_t deviceCount() { return 1; }

ISPCRTDeviceInfo deviceInfo([[maybe_unused]] uint32_t deviceIdx) {
    ISPCRTDeviceInfo info;
    info.deviceId = 0; // for CPU we don't support it yet
    info.vendorId = 0;
    return info;
}

} // namespace cpu

ispcrt::base::MemoryView *CPUDevice::newMemoryView(void *appMem, size_t numBytes,
                                                   const ISPCRTNewMemoryViewFlags *flags) const {
    return new cpu::MemoryView(appMem, numBytes, flags->allocType == ISPCRT_ALLOC_TYPE_SHARED);
}

ispcrt::base::CommandQueue *CPUDevice::newCommandQueue([[maybe_unused]] uint32_t ordinal) const {
    return new cpu::CommandQueueImpl();
}

ispcrt::base::TaskQueue *CPUDevice::newTaskQueue() const { return new cpu::TaskQueue(); }

ispcrt::base::ModuleOptions *CPUDevice::newModuleOptions() const { return new cpu::ModuleOptions(); }

ispcrt::base::ModuleOptions *CPUDevice::newModuleOptions(ISPCRTModuleType moduleType, bool libraryCompilation,
                                                         uint32_t stackSize) const {
    return new cpu::ModuleOptions(moduleType, libraryCompilation, stackSize);
}

ispcrt::base::Module *CPUDevice::newModule(const char *moduleFile,
                                           [[maybe_unused]] const ispcrt::base::ModuleOptions &moduleOpts) const {
    return new cpu::Module(moduleFile);
}

void CPUDevice::dynamicLinkModules([[maybe_unused]] base::Module **modules,
                                   [[maybe_unused]] const uint32_t numModules) const {}

ispcrt::base::Module *CPUDevice::staticLinkModules(base::Module **modules, const uint32_t numModules) const {
    return new cpu::Module((cpu::Module **)modules, numModules);
}

ispcrt::base::Kernel *CPUDevice::newKernel(const ispcrt::base::Module &module, const char *name) const {
    return new cpu::Kernel(module, name);
}

void *CPUDevice::platformNativeHandle() const { return nullptr; }

void *CPUDevice::deviceNativeHandle() const { return nullptr; }

void *CPUDevice::contextNativeHandle() const { return nullptr; }

ISPCRTDeviceType CPUDevice::getType() const { return ISPCRT_DEVICE_TYPE_CPU; }

ISPCRTAllocationType CPUDevice::getMemAllocType([[maybe_unused]] void *appMemory) const {
    return ISPCRT_ALLOC_TYPE_UNKNOWN;
}

ispcrt::base::MemoryView *CPUContext::newMemoryView(void *appMem, size_t numBytes,
                                                    const ISPCRTNewMemoryViewFlags *flags) const {
    return new cpu::MemoryView(appMem, numBytes, flags->allocType == ISPCRT_ALLOC_TYPE_SHARED);
}

ISPCRTDeviceType CPUContext::getDeviceType() const { return ISPCRTDeviceType::ISPCRT_DEVICE_TYPE_CPU; }

void *CPUContext::contextNativeHandle() const { return nullptr; }

} // namespace ispcrt
