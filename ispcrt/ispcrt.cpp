// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#include "ispcrt.h"
// std
#include <exception>
#include <iostream>
#include <string>
// ispcrt
#include "detail/Exception.h"
#include "detail/Module.h"
#include "detail/ModuleOptions.h"
#include "detail/TaskQueue.h"

#ifdef ISPCRT_BUILD_CPU
#include "detail/cpu/CPUContext.h"
#include "detail/cpu/CPUDevice.h"
#endif

#ifdef ISPCRT_BUILD_GPU
#include "detail/gpu/GPUContext.h"
#include "detail/gpu/GPUDevice.h"
#endif

static void defaultErrorFcn(ISPCRTError e, const char *msg) {
    std::cerr << "ISPCRT Error (" << e << "): " << msg << std::endl;
    exit(-1);
}

static ISPCRTErrorFunc g_errorFcn = &defaultErrorFcn;

// Helper functions ///////////////////////////////////////////////////////////

static void handleError(ISPCRTError e, const char *msg) {
    if (g_errorFcn)
        g_errorFcn(e, msg);
}

template <typename OBJECT_T = ispcrt::RefCounted, typename HANDLE_T = ISPCRTGenericHandle>
static OBJECT_T &referenceFromHandle(HANDLE_T handle) {
    return *((OBJECT_T *)handle);
}

#define ISPCRT_CATCH_BEGIN try {
#define ISPCRT_CATCH_END_NO_RETURN()                                                                                   \
    }                                                                                                                  \
    catch (const ispcrt::base::ispcrt_runtime_error &e) {                                                              \
        handleError(e.e, e.what());                                                                                    \
        return;                                                                                                        \
    }                                                                                                                  \
    catch (const std::logic_error &e) {                                                                                \
        handleError(ISPCRT_INVALID_OPERATION, e.what());                                                               \
        return;                                                                                                        \
    }                                                                                                                  \
    catch (const std::exception &e) {                                                                                  \
        handleError(ISPCRT_UNKNOWN_ERROR, e.what());                                                                   \
        return;                                                                                                        \
    }                                                                                                                  \
    catch (...) {                                                                                                      \
        handleError(ISPCRT_UNKNOWN_ERROR, "an unrecognized exception was caught");                                     \
        return;                                                                                                        \
    }

#define ISPCRT_CATCH_END(a)                                                                                            \
    }                                                                                                                  \
    catch (const ispcrt::base::ispcrt_runtime_error &e) {                                                              \
        handleError(e.e, e.what());                                                                                    \
        return a;                                                                                                      \
    }                                                                                                                  \
    catch (const std::logic_error &e) {                                                                                \
        handleError(ISPCRT_INVALID_OPERATION, e.what());                                                               \
        return a;                                                                                                      \
    }                                                                                                                  \
    catch (const std::exception &e) {                                                                                  \
        handleError(ISPCRT_UNKNOWN_ERROR, e.what());                                                                   \
        return a;                                                                                                      \
    }                                                                                                                  \
    catch (...) {                                                                                                      \
        handleError(ISPCRT_UNKNOWN_ERROR, "an unrecognized exception was caught");                                     \
        return a;                                                                                                      \
    }

// Define names of devices libraries.
#if defined(_WIN32) || defined(_WIN64)
#define ISPCRT_SO_LIB_PREFIX ""
#define ISPCRT_SO_LIB_SUFFIX "dll"
#elif defined(__APPLE__)
#define ISPCRT_SO_LIB_PREFIX "lib"
#define ISPCRT_SO_LIB_SUFFIX "dylib"
#else
#define ISPCRT_SO_LIB_PREFIX "lib"
#define ISPCRT_SO_LIB_SUFFIX "so"
#endif

#define ISPCRT_DEVICE_CPU_SOLIB_PREFIX ISPCRT_SO_LIB_PREFIX "ispcrt_device_cpu."
#define ISPCRT_DEVICE_GPU_SOLIB_PREFIX ISPCRT_SO_LIB_PREFIX "ispcrt_device_gpu."
#define ISPCRT_DEVICE_CPU_SOLIB_NAME ISPCRT_DEVICE_CPU_SOLIB_PREFIX ISPCRT_SO_LIB_SUFFIX
#define ISPCRT_DEVICE_GPU_SOLIB_NAME ISPCRT_DEVICE_GPU_SOLIB_PREFIX ISPCRT_SO_LIB_SUFFIX

#if defined(_WIN32) || defined(_WIN64)
#define ISPCRT_DEVICE_CPU_SOLIB_MAJOR_VERSION_NAME nullptr
#define ISPCRT_DEVICE_GPU_SOLIB_MAJOR_VERSION_NAME nullptr
#define ISPCRT_DEVICE_CPU_SOLIB_FULL_VERSION_NAME nullptr
#define ISPCRT_DEVICE_GPU_SOLIB_FULL_VERSION_NAME nullptr
#elif defined(__APPLE__)
#define ISPCRT_DEVICE_CPU_SOLIB_MAJOR_VERSION_NAME                                                                     \
    ISPCRT_DEVICE_CPU_SOLIB_PREFIX "." ISPCRT_VERSION_MAJOR "." ISPCRT_SO_LIB_SUFFIX
#define ISPCRT_DEVICE_GPU_SOLIB_MAJOR_VERSION_NAME                                                                     \
    ISPCRT_DEVICE_GPU_SOLIB_PREFIX "." ISPCRT_VERSION_MAJOR "." ISPCRT_SO_LIB_SUFFIX
#define ISPCRT_DEVICE_CPU_SOLIB_FULL_VERSION_NAME                                                                      \
    ISPCRT_DEVICE_CPU_SOLIB_PREFIX "." ISPCRT_VERSION_FULL "." ISPCRT_SO_LIB_SUFFIX
#define ISPCRT_DEVICE_GPU_SOLIB_FULL_VERSION_NAME                                                                      \
    ISPCRT_DEVICE_GPU_SOLIB_PREFIX "." ISPCRT_VERSION_FULL "." ISPCRT_SO_LIB_SUFFIX
#else
#define ISPCRT_DEVICE_CPU_SOLIB_MAJOR_VERSION_NAME ISPCRT_DEVICE_CPU_SOLIB_NAME "." ISPCRT_VERSION_MAJOR
#define ISPCRT_DEVICE_GPU_SOLIB_MAJOR_VERSION_NAME ISPCRT_DEVICE_GPU_SOLIB_NAME "." ISPCRT_VERSION_MAJOR
#define ISPCRT_DEVICE_CPU_SOLIB_FULL_VERSION_NAME ISPCRT_DEVICE_CPU_SOLIB_NAME "." ISPCRT_VERSION_FULL
#define ISPCRT_DEVICE_GPU_SOLIB_FULL_VERSION_NAME ISPCRT_DEVICE_GPU_SOLIB_NAME "." ISPCRT_VERSION_FULL
#endif

// OS agnostic function to dynamically load a shared library.
void *dyn_load_lib(const char *name, [[maybe_unused]] const char *name_major_version,
                   [[maybe_unused]] const char *name_full_version) {
#if defined(_WIN32) || defined(_WIN64)
    // Removes CWD from the search path to reduce the risk of DLL injection.
    SetDllDirectory("");
    return LoadLibraryEx(name, NULL, 0);
#else
    // Try to load a device library starting from the most specific name down to more general one.
    void *handle = dlopen(name_full_version, RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        return handle;
    }
    handle = dlopen(name_major_version, RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        return handle;
    }
    // This is the most general name, e.g., libispcrt_device_cpu.so. Under some
    // circumstances, it may points to older or newer version. This is probably
    // not a big deal until incompatible changes are encountered between versions.
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

// OS agnostic function to get an address of symbol from the previously loaded shared library.
void *dyn_load_sym(void *handle, const char *symbol) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)GetProcAddress((HMODULE)handle, symbol);
#else
    return dlsym(handle, symbol);
#endif
}

// CPU device API.
static ISPCRTTaskingLaunchFType ispc_launch_fptr = nullptr;
static ISPCRTTaskingAllocFType ispc_alloc_fptr = nullptr;
static ISPCRTTaskingSyncFType ispc_sync_fptr = nullptr;

// Applications can provide their own implementation of ISPCLaunch/ISPCAlloc/ISPCSync tasking API.
void ispcrtSetTaskingCallbacks(ISPCRTTaskingLaunchFType launch, ISPCRTTaskingAllocFType alloc,
                               ISPCRTTaskingSyncFType sync) {
    ispc_launch_fptr = launch;
    ispc_alloc_fptr = alloc;
    ispc_sync_fptr = sync;
}

#ifndef ISPCRT_BUILD_STATIC
// During static linking this API is delivered by linking the real implementation.
// ispc expects these functions to have C linkage / not be mangled
// Stubs to load and call actual implementations (*_cpu) from detail/cpu/ispc_tasking.cpp.
extern "C" {
void ISPCLaunch(void **handlePtr, void *f, void *data, int countx, int county, int countz) {
    if (ispc_launch_fptr) {
        return ispc_launch_fptr(handlePtr, f, data, countx, county, countz);
    }
    // TODO: this code can be called directly from ISPC code. Not sure how to exit error safely.
    fprintf(stderr, "Missing ISPCLaunch symbol");
    abort();
}

void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment) {
    if (ispc_alloc_fptr) {
        return ispc_alloc_fptr(handlePtr, size, alignment);
    }
    // TODO: this code can be called directly from ISPC code. Not sure how to exit error safely.
    fprintf(stderr, "Missing ISPCAlloc symbol");
    abort();
}

void ISPCSync(void *handle) {
    if (ispc_sync_fptr) {
        return ispc_sync_fptr(handle);
    }
    // TODO: this code can be called directly from ISPC code. Not sure how to exit error safely.
    fprintf(stderr, "Missing ISPCSync symbol");
    abort();
}
}
#endif

// Auxiliary function to load device CPU solib and initialize tasking API pointers if build with tasking support.
void *handleCPUDeviceLib() {
    static void *handle = nullptr;
    if (handle) {
        return handle;
    }
    handle = dyn_load_lib(ISPCRT_DEVICE_CPU_SOLIB_NAME, ISPCRT_DEVICE_CPU_SOLIB_MAJOR_VERSION_NAME,
                          ISPCRT_DEVICE_CPU_SOLIB_FULL_VERSION_NAME);
    if (!handle) {
        throw std::runtime_error("Fail to load " ISPCRT_DEVICE_CPU_SOLIB_NAME " library");
    }

#ifdef ISPCRT_BUILD_TASKING
    // Pointers to ISPCLaunch/ISPCAlloc/ISPCSync tasking API may be already initialized by ispcSetTaskingCallbacks.
    if (!ispc_launch_fptr) {
        ispc_launch_fptr = (ISPCRTTaskingLaunchFType)dyn_load_sym(handle, "ISPCLaunch_cpu");
        if (!ispc_launch_fptr) {
            throw std::runtime_error("Missing ISPCLaunch_cpu symbol");
        }
    }

    if (!ispc_alloc_fptr) {
        ispc_alloc_fptr = (ISPCRTTaskingAllocFType)dyn_load_sym(handle, "ISPCAlloc_cpu");
        if (!ispc_alloc_fptr) {
            throw std::runtime_error("Missing ISPCAlloc_cpu symbol");
        }
    }

    if (!ispc_sync_fptr) {
        ispc_sync_fptr = (ISPCRTTaskingSyncFType)dyn_load_sym(handle, "ISPCSync_cpu");
        if (!ispc_sync_fptr) {
            throw std::runtime_error("Missing ISPCSync_cpu symbol");
        }
    }
#endif

    return handle;
}

// Auxiliary function to load device GPU solib.
void *handleGPUDeviceLib() {
    static void *handle = nullptr;
    if (handle) {
        return handle;
    }
    handle = dyn_load_lib(ISPCRT_DEVICE_GPU_SOLIB_NAME, ISPCRT_DEVICE_GPU_SOLIB_MAJOR_VERSION_NAME,
                          ISPCRT_DEVICE_GPU_SOLIB_FULL_VERSION_NAME);
    if (!handle) {
        throw std::runtime_error("Fail to load " ISPCRT_DEVICE_GPU_SOLIB_NAME " library");
    }
    return handle;
}

// Stubs around CPU device solibs API.
uint32_t cpuDeviceCount();
ISPCRTDeviceInfo cpuDeviceInfo(uint32_t idx);
ispcrt::base::Device *loadCPUDevice();
ispcrt::base::Context *loadCPUContext();

// Stubs around GPU device solibs API.
uint32_t gpuDeviceCount();
ISPCRTDeviceInfo gpuDeviceInfo(uint32_t idx);
ispcrt::base::Device *loadGPUDevice();
ispcrt::base::Device *loadGPUDevice(void *ctx, void *dev, uint32_t idx);
ispcrt::base::Context *loadGPUContext();
ispcrt::base::Context *loadGPUContext(void *ctx);

// Function pointer types declarations.
typedef uint32_t (*DeviceCountF)();
typedef ISPCRTDeviceInfo (*DeviceInfoF)(uint32_t);
typedef ispcrt::base::Device *(*LoadDeviceF)();
typedef ispcrt::base::Device *(*LoadDeviceCtxF)(void *, void *, uint32_t);
typedef ispcrt::base::Context *(*LoadContextF)();
typedef ispcrt::base::Context *(*LoadContextCtxF)(void *);

// CPU stubs
uint32_t cpuDeviceCount() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_CPU
    return ispcrt::cpu::deviceCount();
#else
    throw std::runtime_error("CPU support not enabled");
#endif
#else
    static DeviceCountF device_count = nullptr;
    if (device_count) {
        return device_count();
    }

    device_count = (DeviceCountF)dyn_load_sym(handleCPUDeviceLib(), "cpu_device_count");
    if (!device_count) {
        throw std::runtime_error("Missing cpu_device_count symbol");
    }
    return device_count();
#endif
}

ISPCRTDeviceInfo cpuDeviceInfo(uint32_t idx) {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_CPU
    return ispcrt::cpu::deviceInfo(idx);
#else
    throw std::runtime_error("CPU support not enabled");
#endif
#else
    static DeviceInfoF device_info = nullptr;
    if (device_info) {
        return device_info(idx);
    }

    device_info = (DeviceInfoF)dyn_load_sym(handleCPUDeviceLib(), "cpu_device_info");
    if (!device_info) {
        throw std::runtime_error("Missing cpu_device_info symbol");
    }
    return device_info(idx);
#endif
}

ispcrt::base::Device *loadCPUDevice() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_CPU
    return new ispcrt::CPUDevice;
#else
    throw std::runtime_error("CPU support not enabled");
#endif
#else
    static LoadDeviceF load_device = nullptr;
    if (load_device) {
        return load_device();
    }

    load_device = (LoadDeviceF)dyn_load_sym(handleCPUDeviceLib(), "load_cpu_device");
    if (!load_device) {
        throw std::runtime_error("Missing load_cpu_device symbol");
    }

    return load_device();
#endif
}

ispcrt::base::Context *loadCPUContext() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_CPU
    return new ispcrt::CPUContext;
#else
    throw std::runtime_error("CPU support not enabled");
#endif
#else
    static LoadContextF load_context = nullptr;
    if (load_context) {
        return load_context();
    }

    load_context = (LoadContextF)dyn_load_sym(handleCPUDeviceLib(), "load_cpu_context");
    if (!load_context) {
        throw std::runtime_error("Missing load_cpu_context symbol");
    }

    return load_context();
#endif
}

// GPU stubs.
uint32_t gpuDeviceCount() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return ispcrt::gpu::deviceCount();
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static DeviceCountF device_count = nullptr;
    if (device_count) {
        return device_count();
    }

    device_count = (DeviceCountF)dyn_load_sym(handleGPUDeviceLib(), "gpu_device_count");
    if (!device_count) {
        throw std::runtime_error("Missing gpu_device_count symbol");
    }
    return device_count();
#endif
}

ISPCRTDeviceInfo gpuDeviceInfo([[maybe_unused]] uint32_t idx) {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return ispcrt::gpu::deviceInfo(idx);
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static DeviceInfoF device_info = nullptr;
    if (device_info) {
        return device_info(idx);
    }

    device_info = (DeviceInfoF)dyn_load_sym(handleGPUDeviceLib(), "gpu_device_info");
    if (!device_info) {
        throw std::runtime_error("Missing gpu_device_info symbol");
    }
    return device_info(idx);
#endif
}

ispcrt::base::Device *loadGPUDevice() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return new ispcrt::GPUDevice;
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static LoadDeviceF load_device = nullptr;
    if (load_device) {
        return load_device();
    }

    load_device = (LoadDeviceF)dyn_load_sym(handleGPUDeviceLib(), "load_gpu_device");
    if (!load_device) {
        throw std::runtime_error("Missing load_gpu_device symbol");
    }

    return load_device();
#endif
}

ispcrt::base::Device *loadGPUDevice([[maybe_unused]] void *ctx, [[maybe_unused]] void *dev,
                                    [[maybe_unused]] uint32_t idx) {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return new ispcrt::GPUDevice(ctx, dev, idx);
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static LoadDeviceCtxF load_device = nullptr;
    if (load_device) {
        return load_device(ctx, dev, idx);
    }

    load_device = (LoadDeviceCtxF)dyn_load_sym(handleGPUDeviceLib(), "load_gpu_device_ctx");
    if (!load_device) {
        throw std::runtime_error("Missing load_gpu_device_ctx symbol");
    }

    return load_device(ctx, dev, idx);
#endif
}

ispcrt::base::Context *loadGPUContext() {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return new ispcrt::GPUContext;
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static LoadContextF load_context = nullptr;
    if (load_context) {
        return load_context();
    }

    load_context = (LoadContextF)dyn_load_sym(handleGPUDeviceLib(), "load_gpu_context");
    if (!load_context) {
        throw std::runtime_error("Missing load_gpu_context symbol");
    }

    return load_context();
#endif
}

ispcrt::base::Context *loadGPUContext([[maybe_unused]] void *ctx) {
#ifdef ISPCRT_BUILD_STATIC
#ifdef ISPCRT_BUILD_GPU
    return new ispcrt::GPUContext(ctx);
#else
    throw std::runtime_error("GPU support not enabled");
#endif
#else
    static LoadContextCtxF load_context = nullptr;
    if (load_context) {
        return load_context(ctx);
    }

    load_context = (LoadContextCtxF)dyn_load_sym(handleGPUDeviceLib(), "load_gpu_context_ctx");
    if (!load_context) {
        throw std::runtime_error("Missing load_gpu_context_ctx symbol");
    }

    return load_context(ctx);
#endif
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////// API DEFINITIONS ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" {

void ispcrtSetErrorFunc(ISPCRTErrorFunc fcn) { g_errorFcn = fcn; }

///////////////////////////////////////////////////////////////////////////////
// Object lifetime ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

long long ispcrtUseCount(ISPCRTGenericHandle h) ISPCRT_CATCH_BEGIN {
    auto &obj = referenceFromHandle(h);
    return obj.useCount();
}
ISPCRT_CATCH_END(0)

void ispcrtRelease(ISPCRTGenericHandle h) ISPCRT_CATCH_BEGIN {
    auto &obj = referenceFromHandle(h);
    obj.refDec();
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtRetain(ISPCRTGenericHandle h) ISPCRT_CATCH_BEGIN {
    auto &obj = referenceFromHandle(h);
    obj.refInc();
}
ISPCRT_CATCH_END_NO_RETURN()

///////////////////////////////////////////////////////////////////////////////
// Device initialization //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
static ISPCRTDevice getISPCRTDevice(ISPCRTDeviceType type, ISPCRTContext context,
                                    [[maybe_unused]] ISPCRTGenericHandle d,
                                    [[maybe_unused]] uint32_t deviceIdx) ISPCRT_CATCH_BEGIN {
    ispcrt::base::Device *device = nullptr;

    [[maybe_unused]] void *nativeContext = nullptr;
    if (context) {
        auto &c = referenceFromHandle<ispcrt::base::Context>(context);
        nativeContext = c.contextNativeHandle();
    }
    switch (type) {
    case ISPCRT_DEVICE_TYPE_AUTO: {
#if defined(ISPCRT_BUILD_GPU) && defined(ISPCRT_BUILD_CPU)
        try {
            device = loadGPUDevice();
        } catch (...) {
            if (device)
                delete device;
            device = loadCPUDevice();
        }
#elif defined(ISPCRT_BUILD_CPU)
        device = loadCPUDevice();
#elif defined(ISPCRT_BUILD_GPU)
        device = loadGPUDevice();
#endif
        break;
    }
    case ISPCRT_DEVICE_TYPE_GPU:
#ifdef ISPCRT_BUILD_GPU
        device = loadGPUDevice(nativeContext, d, deviceIdx);
#else
        throw std::runtime_error("GPU support not enabled");
#endif
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
#ifdef ISPCRT_BUILD_CPU
        device = loadCPUDevice();
#else
        throw std::runtime_error("CPU support not enabled");
#endif
        break;
    default:
        throw std::runtime_error("Unknown device type queried!");
    }

    return (ISPCRTDevice)device;
}
ISPCRT_CATCH_END(0)

ISPCRTDevice ispcrtGetDevice(ISPCRTDeviceType type, uint32_t deviceIdx) {
    return getISPCRTDevice(type, nullptr, nullptr, deviceIdx);
}

ISPCRTDevice ispcrtGetDeviceFromContext(ISPCRTContext context, uint32_t deviceIdx) {
    auto &c = referenceFromHandle<ispcrt::base::Context>(context);
    return getISPCRTDevice(c.getDeviceType(), context, nullptr, deviceIdx);
}

ISPCRTDevice ispcrtGetDeviceFromNativeHandle(ISPCRTContext context, ISPCRTGenericHandle d) {
    auto &c = referenceFromHandle<ispcrt::base::Context>(context);
    return getISPCRTDevice(c.getDeviceType(), context, d, 0);
}

ISPCRTDeviceType ispcrtGetDeviceType(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return device.getType();
}
ISPCRT_CATCH_END(ISPCRT_DEVICE_TYPE_AUTO)

uint32_t ispcrtGetDeviceCount(ISPCRTDeviceType type) ISPCRT_CATCH_BEGIN {
    uint32_t devices = 0;

    switch (type) {
    case ISPCRT_DEVICE_TYPE_AUTO:
        throw std::runtime_error("Device type must be specified");
        break;
    case ISPCRT_DEVICE_TYPE_GPU:
#ifdef ISPCRT_BUILD_GPU
        devices = gpuDeviceCount();
#else
        throw std::runtime_error("GPU support not enabled");
#endif
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
#ifdef ISPCRT_BUILD_CPU
        devices = cpuDeviceCount();
#else
        throw std::runtime_error("CPU support not enabled");
#endif
        break;
    default:
        throw std::runtime_error("Unknown device type queried!");
    }

    return devices;
}
ISPCRT_CATCH_END(0)

void ispcrtGetDeviceInfo(ISPCRTDeviceType type, uint32_t deviceIdx, ISPCRTDeviceInfo *info) ISPCRT_CATCH_BEGIN {
    if (info == nullptr)
        throw std::runtime_error("info cannot be null!");

    switch (type) {
    case ISPCRT_DEVICE_TYPE_AUTO:
        throw std::runtime_error("Device type must be specified");
        break;
    case ISPCRT_DEVICE_TYPE_GPU:
#ifdef ISPCRT_BUILD_GPU
        *info = gpuDeviceInfo(deviceIdx);
#else
        throw std::runtime_error("GPU support not enabled");
#endif
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
#ifdef ISPCRT_BUILD_CPU
        *info = cpuDeviceInfo(deviceIdx);
#else
        throw std::runtime_error("CPU support not enabled");
#endif
        break;
    default:
        throw std::runtime_error("Unknown device type queried!");
    }
}
ISPCRT_CATCH_END_NO_RETURN()

///////////////////////////////////////////////////////////////////////////////
// Context initialization //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
static ISPCRTContext getISPCRTContext(ISPCRTDeviceType type,
                                      [[maybe_unused]] ISPCRTGenericHandle c) ISPCRT_CATCH_BEGIN {
    ispcrt::base::Context *context = nullptr;

    switch (type) {
    case ISPCRT_DEVICE_TYPE_AUTO: {
#if defined(ISPCRT_BUILD_GPU) && defined(ISPCRT_BUILD_CPU)
        try {
            context = loadGPUContext();
        } catch (...) {
            if (context)
                delete context;
            context = loadCPUContext();
        }
#elif defined(ISPCRT_BUILD_CPU)
        context = loadCPUContext();
#elif defined(ISPCRT_BUILD_GPU)
        context = loadGPUContext();
#endif
        break;
    }
    case ISPCRT_DEVICE_TYPE_GPU:
#ifdef ISPCRT_BUILD_GPU
        context = loadGPUContext(c);
#else
        throw std::runtime_error("GPU support not enabled");
#endif
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
#ifdef ISPCRT_BUILD_CPU
        context = loadCPUContext();
#else
        throw std::runtime_error("CPU support not enabled");
#endif
        break;
    default:
        throw std::runtime_error("Unknown device type queried!");
    }

    return (ISPCRTContext)context;
}
ISPCRT_CATCH_END(0)

ISPCRTContext ispcrtNewContext(ISPCRTDeviceType type) { return getISPCRTContext(type, nullptr); }

ISPCRTContext ispcrtGetContextFromNativeHandle(ISPCRTDeviceType type, ISPCRTGenericHandle c) {
    return getISPCRTContext(type, c);
}
///////////////////////////////////////////////////////////////////////////////
// MemoryViews ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ISPCRTMemoryView ispcrtNewMemoryView(ISPCRTDevice d, void *appMemory, size_t numBytes,
                                     ISPCRTNewMemoryViewFlags *flags) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    if (flags->allocType != ISPCRT_ALLOC_TYPE_SHARED && flags->allocType != ISPCRT_ALLOC_TYPE_DEVICE) {
        throw std::runtime_error("Unsupported memory allocation type requested!");
    }
    return (ISPCRTMemoryView)device.newMemoryView(appMemory, numBytes, flags);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTMemoryView ispcrtNewMemoryViewForContext(ISPCRTContext c, void *appMemory, size_t numBytes,
                                               ISPCRTNewMemoryViewFlags *flags) ISPCRT_CATCH_BEGIN {
    const auto &context = referenceFromHandle<ispcrt::base::Context>(c);
    if (flags->allocType != ISPCRT_ALLOC_TYPE_SHARED) {
        throw std::runtime_error("Only shared memory allocation is allowed for context!");
    }
    return (ISPCRTMemoryView)context.newMemoryView(appMemory, numBytes, flags);
}
ISPCRT_CATCH_END(nullptr)

void *ispcrtHostPtr(ISPCRTMemoryView h) ISPCRT_CATCH_BEGIN {
    auto &mv = referenceFromHandle<ispcrt::base::MemoryView>(h);
    return mv.hostPtr();
}
ISPCRT_CATCH_END(nullptr)

void *ispcrtDevicePtr(ISPCRTMemoryView h) ISPCRT_CATCH_BEGIN {
    auto &mv = referenceFromHandle<ispcrt::base::MemoryView>(h);
    return mv.devicePtr();
}
ISPCRT_CATCH_END(nullptr)

size_t ispcrtSize(ISPCRTMemoryView h) ISPCRT_CATCH_BEGIN {
    auto &mv = referenceFromHandle<ispcrt::base::MemoryView>(h);
    return mv.numBytes();
}
ISPCRT_CATCH_END(0)

ISPCRTAllocationType ispcrtGetMemoryViewAllocType(ISPCRTMemoryView h) ISPCRT_CATCH_BEGIN {
    auto &mv = referenceFromHandle<ispcrt::base::MemoryView>(h);
    return mv.isShared() ? ISPCRTAllocationType::ISPCRT_ALLOC_TYPE_SHARED
                         : ISPCRTAllocationType::ISPCRT_ALLOC_TYPE_DEVICE;
}
ISPCRT_CATCH_END(ISPCRTAllocationType::ISPCRT_ALLOC_TYPE_UNKNOWN)

ISPCRTAllocationType ispcrtGetMemoryAllocType(ISPCRTDevice d, void *memBuffer) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return device.getMemAllocType(memBuffer);
}
ISPCRT_CATCH_END(ISPCRTAllocationType::ISPCRT_ALLOC_TYPE_UNKNOWN)

void *ispcrtSharedPtr(ISPCRTMemoryView h) ISPCRT_CATCH_BEGIN {
    auto &mv = referenceFromHandle<ispcrt::base::MemoryView>(h);
    return mv.devicePtr();
}
ISPCRT_CATCH_END(nullptr)

///////////////////////////////////////////////////////////////////////////////
// Modules ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ISPCRTModuleOptions ispcrtNewModuleOptionsEmpty(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return (ISPCRTModuleOptions)device.newModuleOptions();
}
ISPCRT_CATCH_END(nullptr)

ISPCRTModuleOptions ispcrtNewModuleOptions(ISPCRTDevice d, ISPCRTModuleType moduleType, bool libraryCompilation,
                                           uint32_t stackSize) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return (ISPCRTModuleOptions)device.newModuleOptions(moduleType, libraryCompilation, stackSize);
}
ISPCRT_CATCH_END(nullptr)

uint32_t ispcrtModuleOptionsGetStackSize(ISPCRTModuleOptions o) ISPCRT_CATCH_BEGIN {
    const auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    return opts.stackSize();
}
ISPCRT_CATCH_END(0)

bool ispcrtModuleOptionsGetLibraryCompilation(ISPCRTModuleOptions o) ISPCRT_CATCH_BEGIN {
    const auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    return opts.libraryCompilation();
}
ISPCRT_CATCH_END(false)

ISPCRTModuleType ispcrtModuleOptionsGetModuleType(ISPCRTModuleOptions o) ISPCRT_CATCH_BEGIN {
    const auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    return opts.moduleType();
}
ISPCRT_CATCH_END(ISPCRTModuleType::ISPCRT_VECTOR_MODULE)

void ispcrtModuleOptionsSetStackSize(ISPCRTModuleOptions o, uint32_t size) ISPCRT_CATCH_BEGIN {
    auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    opts.setStackSize(size);
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtModuleOptionsSetLibraryCompilation(ISPCRTModuleOptions o, bool isLibraryCompilation) ISPCRT_CATCH_BEGIN {
    auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    opts.setLibraryCompilation(isLibraryCompilation);
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtModuleOptionsSetModuleType(ISPCRTModuleOptions o, ISPCRTModuleType type) ISPCRT_CATCH_BEGIN {
    auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    opts.setModuleType(type);
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTModule ispcrtLoadModule(ISPCRTDevice d, const char *moduleFile) ISPCRT_CATCH_BEGIN {
    ISPCRTModule module;
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    const auto &o = (ISPCRTModuleOptions)device.newModuleOptions();
    module = ispcrtLoadModuleWithOptions(d, moduleFile, o);
    ispcrtRelease(o);
    return module;
}
ISPCRT_CATCH_END(nullptr)

ISPCRTModule ispcrtLoadModuleWithOptions(ISPCRTDevice d, const char *moduleFile,
                                         ISPCRTModuleOptions o) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    const auto &opts = referenceFromHandle<ispcrt::base::ModuleOptions>(o);
    return (ISPCRTModule)device.newModule(moduleFile, opts);
}
ISPCRT_CATCH_END(nullptr)

void ispcrtDynamicLinkModules(ISPCRTDevice d, ISPCRTModule *modules, const uint32_t numModules) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    device.dynamicLinkModules((ispcrt::base::Module **)modules, numModules);
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTModule ispcrtStaticLinkModules(ISPCRTDevice d, ISPCRTModule *modules,
                                     const uint32_t numModules) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return (ISPCRTModule)device.staticLinkModules((ispcrt::base::Module **)modules, numModules);
}
ISPCRT_CATCH_END(nullptr)

void *ispcrtFunctionPtr(ISPCRTModule m, const char *name) ISPCRT_CATCH_BEGIN {
    const auto &module = referenceFromHandle<ispcrt::base::Module>(m);
    return module.functionPtr(name);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTKernel ispcrtNewKernel(ISPCRTDevice d, ISPCRTModule m, const char *name) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    const auto &module = referenceFromHandle<ispcrt::base::Module>(m);
    return (ISPCRTKernel)device.newKernel(module, name);
}
ISPCRT_CATCH_END(nullptr)

///////////////////////////////////////////////////////////////////////////////
// Command lists //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void ispcrtCommandListBarrier(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    list.barrier();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTFuture ispcrtCommandListCopyToDevice(ISPCRTCommandList l, ISPCRTMemoryView mv) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    auto &view = referenceFromHandle<ispcrt::base::MemoryView>(mv);
    return (ISPCRTFuture)list.copyToDevice(view);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtCommandListCopyToHost(ISPCRTCommandList l, ISPCRTMemoryView mv) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    auto &view = referenceFromHandle<ispcrt::base::MemoryView>(mv);
    return (ISPCRTFuture)list.copyToHost(view);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtCommandListCopyMemoryView(ISPCRTCommandList l, ISPCRTMemoryView mvDst, ISPCRTMemoryView mvSrc,
                                             const size_t size) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    auto &viewDst = referenceFromHandle<ispcrt::base::MemoryView>(mvDst);
    auto &viewSrc = referenceFromHandle<ispcrt::base::MemoryView>(mvSrc);
    if (size > viewDst.numBytes()) {
        throw std::runtime_error("Requested copy size is bigger than destination buffer size!");
    }
    if (size > viewSrc.numBytes()) {
        throw std::runtime_error("Requested copy size is bigger than source buffer size!");
    }
    return (ISPCRTFuture)list.copyMemoryView(viewDst, viewSrc, size);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtCommandListLaunch1D(ISPCRTCommandList l, ISPCRTKernel k, ISPCRTMemoryView p,
                                       size_t dim0) ISPCRT_CATCH_BEGIN {
    return ispcrtCommandListLaunch3D(l, k, p, dim0, 1, 1);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtCommandListLaunch2D(ISPCRTCommandList l, ISPCRTKernel k, ISPCRTMemoryView p, size_t dim0,
                                       size_t dim1) ISPCRT_CATCH_BEGIN {
    return ispcrtCommandListLaunch3D(l, k, p, dim0, dim1, 1);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtCommandListLaunch3D(ISPCRTCommandList l, ISPCRTKernel k, ISPCRTMemoryView p, size_t dim0,
                                       size_t dim1, size_t dim2) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    auto &kernel = referenceFromHandle<ispcrt::base::Kernel>(k);

    ispcrt::base::MemoryView *params = nullptr;

    if (p)
        params = &referenceFromHandle<ispcrt::base::MemoryView>(p);

    return (ISPCRTFuture)list.launch(kernel, params, dim0, dim1, dim2);
}
ISPCRT_CATCH_END(nullptr)

void ispcrtCommandListClose(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    list.close();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTFence ispcrtCommandListSubmit(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    return (ISPCRTFence)list.submit();
}
ISPCRT_CATCH_END(nullptr)

void ispcrtCommandListReset(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    list.reset();
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtCommandListEnableTimestamps(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    return list.enableTimestamps();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTGenericHandle ispcrtCommandListNativeHandle(ISPCRTCommandList l) ISPCRT_CATCH_BEGIN {
    auto &list = referenceFromHandle<ispcrt::base::CommandList>(l);
    return list.nativeHandle();
}
ISPCRT_CATCH_END(nullptr)

///////////////////////////////////////////////////////////////////////////////
// Command queues /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ISPCRTCommandQueue ispcrtNewCommandQueue(ISPCRTDevice d, uint32_t ordinal) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return (ISPCRTCommandQueue)device.newCommandQueue(ordinal);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTCommandList ispcrtCommandQueueCreateCommandList(ISPCRTCommandQueue q) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::CommandQueue>(q);
    return (ISPCRTCommandList)queue.createCommandList();
}
ISPCRT_CATCH_END(nullptr)

void ispcrtCommandQueueSync(ISPCRTCommandQueue q) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::CommandQueue>(q);
    queue.sync();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTGenericHandle ispcrtCommandQueueNativeHandle(ISPCRTCommandQueue q) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::CommandQueue>(q);
    return queue.nativeHandle();
}
ISPCRT_CATCH_END(nullptr)

///////////////////////////////////////////////////////////////////////////////
// Task queues ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ISPCRTTaskQueue ispcrtNewTaskQueue(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return (ISPCRTTaskQueue)device.newTaskQueue();
}
ISPCRT_CATCH_END(nullptr)

void ispcrtDeviceBarrier(ISPCRTTaskQueue q) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    queue.barrier();
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtCopyToDevice(ISPCRTTaskQueue q, ISPCRTMemoryView mv) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    auto &view = referenceFromHandle<ispcrt::base::MemoryView>(mv);
    queue.copyToDevice(view);
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtCopyToHost(ISPCRTTaskQueue q, ISPCRTMemoryView mv) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    auto &view = referenceFromHandle<ispcrt::base::MemoryView>(mv);
    queue.copyToHost(view);
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtCopyMemoryView(ISPCRTTaskQueue q, ISPCRTMemoryView mvDst, ISPCRTMemoryView mvSrc,
                          const size_t size) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    auto &viewDst = referenceFromHandle<ispcrt::base::MemoryView>(mvDst);
    auto &viewSrc = referenceFromHandle<ispcrt::base::MemoryView>(mvSrc);
    if (size > viewDst.numBytes()) {
        throw std::runtime_error("Requested copy size is bigger than destination buffer size!");
    }
    if (size > viewSrc.numBytes()) {
        throw std::runtime_error("Requested copy size is bigger than source buffer size!");
    }
    queue.copyMemoryView(viewDst, viewSrc, size);
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTFuture ispcrtLaunch1D(ISPCRTTaskQueue q, ISPCRTKernel k, ISPCRTMemoryView p, size_t dim0) ISPCRT_CATCH_BEGIN {
    return ispcrtLaunch3D(q, k, p, dim0, 1, 1);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtLaunch2D(ISPCRTTaskQueue q, ISPCRTKernel k, ISPCRTMemoryView p, size_t dim0,
                            size_t dim1) ISPCRT_CATCH_BEGIN {
    return ispcrtLaunch3D(q, k, p, dim0, dim1, 1);
}
ISPCRT_CATCH_END(nullptr)

ISPCRTFuture ispcrtLaunch3D(ISPCRTTaskQueue q, ISPCRTKernel k, ISPCRTMemoryView p, size_t dim0, size_t dim1,
                            size_t dim2) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    auto &kernel = referenceFromHandle<ispcrt::base::Kernel>(k);

    ispcrt::base::MemoryView *params = nullptr;

    if (p)
        params = &referenceFromHandle<ispcrt::base::MemoryView>(p);

    return (ISPCRTFuture)queue.launch(kernel, params, dim0, dim1, dim2);
}
ISPCRT_CATCH_END(nullptr)

void ispcrtSync(ISPCRTTaskQueue q) ISPCRT_CATCH_BEGIN {
    auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    queue.sync();
}
ISPCRT_CATCH_END_NO_RETURN()

void ispcrtFenceSync(ISPCRTFence f) ISPCRT_CATCH_BEGIN {
    auto &fence = referenceFromHandle<ispcrt::base::Fence>(f);
    fence.sync();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTFenceStatus ispcrtFenceStatus(ISPCRTFence f) ISPCRT_CATCH_BEGIN {
    auto &fence = referenceFromHandle<ispcrt::base::Fence>(f);
    return fence.status();
}
ISPCRT_CATCH_END(ISPCRT_FENCE_SIGNALED)

void ispcrtFenceReset(ISPCRTFence f) ISPCRT_CATCH_BEGIN {
    auto &fence = referenceFromHandle<ispcrt::base::Fence>(f);
    fence.reset();
}
ISPCRT_CATCH_END_NO_RETURN()

ISPCRTGenericHandle ispcrtFenceNativeHandle(ISPCRTFence f) ISPCRT_CATCH_BEGIN {
    auto &fence = referenceFromHandle<ispcrt::base::Fence>(f);
    return fence.nativeHandle();
}
ISPCRT_CATCH_END(nullptr)

uint64_t ispcrtFutureGetTimeNs(ISPCRTFuture f) ISPCRT_CATCH_BEGIN {
    if (!f)
        return -1;

    auto &future = referenceFromHandle<ispcrt::base::Future>(f);

    if (!future.valid())
        return -1;

    return future.time();
}
ISPCRT_CATCH_END(-1)

bool ispcrtFutureIsValid(ISPCRTFuture f) ISPCRT_CATCH_BEGIN {
    auto &future = referenceFromHandle<ispcrt::base::Future>(f);
    return future.valid();
}
ISPCRT_CATCH_END(false)

///////////////////////////////////////////////////////////////////////////////
// Native handles//////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ISPCRTGenericHandle ispcrtPlatformNativeHandle(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return device.platformNativeHandle();
}
ISPCRT_CATCH_END(nullptr)

ISPCRTGenericHandle ispcrtDeviceNativeHandle(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return device.deviceNativeHandle();
}
ISPCRT_CATCH_END(nullptr)

ISPCRTGenericHandle ispcrtDeviceContextNativeHandle(ISPCRTDevice d) ISPCRT_CATCH_BEGIN {
    const auto &device = referenceFromHandle<ispcrt::base::Device>(d);
    return device.contextNativeHandle();
}
ISPCRT_CATCH_END(nullptr)

ISPCRTGenericHandle ispcrtContextNativeHandle(ISPCRTContext c) ISPCRT_CATCH_BEGIN {
    const auto &context = referenceFromHandle<ispcrt::base::Context>(c);
    return context.contextNativeHandle();
}
ISPCRT_CATCH_END(nullptr)

ISPCRTGenericHandle ispcrtTaskQueueNativeHandle(ISPCRTTaskQueue q) ISPCRT_CATCH_BEGIN {
    const auto &queue = referenceFromHandle<ispcrt::base::TaskQueue>(q);
    return queue.taskQueueNativeHandle();
}
ISPCRT_CATCH_END(nullptr)
} // extern "C"
