// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "CPUDevice.h"

#ifdef _WIN32
#error "Windows not yet supported!"
#else
#include <dlfcn.h>
#endif
// std
#include <exception>
#include <string>

namespace ispcrt {
namespace cpu {

using CPUKernelEntryPoint = void (*)(void *, size_t, size_t, size_t);

struct MemoryView : public ispcrt::MemoryView {
    MemoryView(void *appMem, size_t numBytes) : m_mem(appMem), m_size(numBytes) {}

    void *hostPtr() { return m_mem; };

    void *devicePtr() { return m_mem; };

    size_t numBytes() { return m_size; };

  private:
    void *m_mem{nullptr};
    size_t m_size{0};
};

struct Module : public ispcrt::Module {
    Module(const char *moduleFile) : m_file(moduleFile) {
        if (!m_file.empty()) {
            m_lib = dlopen(("lib" + m_file + ".so").c_str(), RTLD_LAZY | RTLD_LOCAL);

            if (!m_lib)
                throw std::logic_error("could not open CPU shared module file");
        }
    }

    ~Module() {
        if (m_lib)
            dlclose(m_lib);
    }

    void *lib() const { return m_lib; }

  private:
    std::string m_file;
    void *m_lib{nullptr};
};

struct Kernel : public ispcrt::Kernel {
    Kernel(const ispcrt::Module &_module, const char *_name) : m_fcnName(_name), m_module(&_module) {
        const cpu::Module &module = (const cpu::Module &)_module;

        auto name = std::string(_name) + "_cpu_entry_point";

        void *fcn = dlsym(module.lib() ? module.lib() : RTLD_DEFAULT, name.c_str());

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

    const ispcrt::Module *m_module{nullptr};
};

struct TaskQueue : public ispcrt::TaskQueue {
    TaskQueue() {
        // no-op
    }

    void barrier() override {
        // no-op
    }

    void copyToHost(ispcrt::MemoryView &) override {
        // no-op
    }

    void copyToDevice(ispcrt::MemoryView &) override {
        // no-op
    }

    void launch(ispcrt::Kernel &k, ispcrt::MemoryView *params, size_t dim0, size_t dim1, size_t dim2) override {
        auto &kernel = (cpu::Kernel &)k;
        auto *parameters = (cpu::MemoryView *)params;

        auto *fcn = kernel.entryPoint();

        fcn(parameters ? parameters->devicePtr() : nullptr, dim0, dim1, dim2);
    }

    void sync() override {
        // no-op
    }
};
} // namespace cpu

MemoryView *CPUDevice::newMemoryView(void *appMem, size_t numBytes) const {
    return new cpu::MemoryView(appMem, numBytes);
}

TaskQueue *CPUDevice::newTaskQueue() const { return new cpu::TaskQueue(); }

Module *CPUDevice::newModule(const char *moduleFile) const { return new cpu::Module(moduleFile); }

Kernel *CPUDevice::newKernel(const Module &module, const char *name) const { return new cpu::Kernel(module, name); }

} // namespace ispcrt
