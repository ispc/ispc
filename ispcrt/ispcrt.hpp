// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ispcrt.h"
// std
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace ispcrt {

/////////////////////////////////////////////////////////////////////////////
// Generic base handle wrapper //////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename HANDLE_T = ISPCRTGenericHandle> class GenericObject {
  public:
    GenericObject(HANDLE_T object = nullptr);
    virtual ~GenericObject();

    GenericObject(const GenericObject<HANDLE_T> &copy);
    GenericObject(GenericObject<HANDLE_T> &&move);

    GenericObject &operator=(const GenericObject<HANDLE_T> &copy);
    GenericObject &operator=(GenericObject<HANDLE_T> &&move);

    HANDLE_T handle() const;

    operator bool() const;

  protected:
    void setHandle(HANDLE_T object);

    HANDLE_T m_handle{nullptr};
};

// Inlined definitions //

template <typename HANDLE_T> inline GenericObject<HANDLE_T>::GenericObject(HANDLE_T object) : m_handle(object) {}

template <typename HANDLE_T> inline GenericObject<HANDLE_T>::~GenericObject() {
    if (m_handle)
        ispcrtRelease(m_handle);
}

template <typename HANDLE_T> inline GenericObject<HANDLE_T>::GenericObject(const GenericObject<HANDLE_T> &copy) {
    m_handle = copy.handle();
    ispcrtRetain(copy.handle());
}

template <typename HANDLE_T> inline GenericObject<HANDLE_T>::GenericObject(GenericObject<HANDLE_T> &&move) {
    m_handle = move.handle();
    move.m_handle = nullptr;
}

template <typename HANDLE_T>
inline GenericObject<HANDLE_T> &GenericObject<HANDLE_T>::operator=(const GenericObject<HANDLE_T> &copy) {
    m_handle = copy.handle();
    ispcrtRetain(copy.handle());
    return *this;
}

template <typename HANDLE_T>
inline GenericObject<HANDLE_T> &GenericObject<HANDLE_T>::operator=(GenericObject<HANDLE_T> &&move) {
    m_handle = move.handle();
    move.m_handle = nullptr;
    return *this;
}

template <typename HANDLE_T> inline HANDLE_T GenericObject<HANDLE_T>::handle() const { return m_handle; }

template <typename HANDLE_T> inline GenericObject<HANDLE_T>::operator bool() const { return handle() != nullptr; }

template <typename HANDLE_T> inline void GenericObject<HANDLE_T>::setHandle(HANDLE_T object) {
    m_handle = object;
    ispcrtRetain(m_handle);
}

/////////////////////////////////////////////////////////////////////////////
// Future wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Future : public GenericObject<ISPCRTFuture> {
  public:
    Future() = default;
    Future(ISPCRTFuture f);
    ~Future() = default;
    bool valid() const;
    uint64_t time() const;
};

inline Future::Future(ISPCRTFuture f) : GenericObject<ISPCRTFuture>(f) {
    if (f)
        ispcrtRetain(f);
}

inline bool Future::valid() const { return handle() && ispcrtFutureIsValid(handle()); }

inline uint64_t Future::time() const { return ispcrtFutureGetTimeNs(handle()); }

/////////////////////////////////////////////////////////////////////////////
// Fence wrapper ////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Fence : public GenericObject<ISPCRTFence> {
  public:
    Fence() = default;
    Fence(ISPCRTFence f);
    ~Fence() = default;
    void sync();
    ISPCRTFenceStatus status() const;
    void reset();
    void *nativeFenceHandle() const;
};

inline Fence::Fence(ISPCRTFence f) : GenericObject<ISPCRTFence>(f) {
    if (f)
        ispcrtRetain(f);
}

inline void Fence::sync() { ispcrtFenceSync(handle()); }

inline ISPCRTFenceStatus Fence::status() const { return ispcrtFenceStatus(handle()); }

inline void Fence::reset() { ispcrtFenceReset(handle()); }

inline void *Fence::nativeFenceHandle() const { return ispcrtFenceNativeHandle(handle()); }

/////////////////////////////////////////////////////////////////////////////
// Context wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Context : public GenericObject<ISPCRTContext> {
  public:
    Context() = default;
    Context(ISPCRTDeviceType type);
    Context(ISPCRTDeviceType type, ISPCRTGenericHandle nativeContextHandle);
    ~Context() = default;
    void *nativeContextHandle() const;
};

// Inlined definitions //
inline Context::Context(ISPCRTDeviceType type) : GenericObject<ISPCRTContext>(ispcrtNewContext(type)) {}

inline Context::Context(ISPCRTDeviceType type, ISPCRTGenericHandle nativeContextHandle)
    : GenericObject<ISPCRTContext>(ispcrtGetContextFromNativeHandle(type, nativeContextHandle)) {}

inline void *Context::nativeContextHandle() const { return ispcrtContextNativeHandle(handle()); }
/////////////////////////////////////////////////////////////////////////////
// Device wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
class Module;
class ModuleOptions;
class Device : public GenericObject<ISPCRTDevice> {
  public:
    Device() = default;
    Device(ISPCRTDeviceType type);
    Device(const Context &context);
    // deviceIdx is an index of the device in the list of supported devices
    // The list of the supported devices can be obtained with:
    // - allDevicesInformation() call or
    // - deviceCount() call and a series of deviceInformation() calls
    Device(ISPCRTDeviceType type, uint32_t deviceIdx);
    Device(const Context &context, uint32_t deviceIdx);
    Device(const Context &context, ISPCRTGenericHandle nativeDeviceHandle);
    void *nativePlatformHandle() const;
    void *nativeDeviceHandle() const;
    void *nativeContextHandle() const;
    ISPCRTDeviceType getType() const;
    // static methods to get information about available devices
    static uint32_t deviceCount(ISPCRTDeviceType type);
    static ISPCRTDeviceInfo deviceInformation(ISPCRTDeviceType type, uint32_t deviceIdx);
    static std::vector<ISPCRTDeviceInfo> allDevicesInformation(ISPCRTDeviceType type);
    // link modules
    void dynamicLinkModules(ISPCRTModule *modules, const uint32_t num);
    Module staticLinkModules(ISPCRTModule *modules, const uint32_t num);
    // check memory type
    ISPCRTAllocationType getMemoryAllocType(void *memBuffer);
};

// Inlined definitions //
inline Device::Device(ISPCRTDeviceType type, uint32_t deviceIdx)
    : GenericObject<ISPCRTDevice>(ispcrtGetDevice(type, deviceIdx)) {}
inline Device::Device(ISPCRTDeviceType type) : Device(type, uint32_t(0)) {}

inline Device::Device(const Context &context, uint32_t deviceIdx)
    : GenericObject<ISPCRTDevice>(ispcrtGetDeviceFromContext(context.handle(), deviceIdx)) {}
inline Device::Device(const Context &context) : Device(context, uint32_t(0)) {}

inline Device::Device(const Context &context, ISPCRTGenericHandle nativeDeviceHandle)
    : GenericObject<ISPCRTDevice>(ispcrtGetDeviceFromNativeHandle(context.handle(), nativeDeviceHandle)) {}

inline void *Device::nativePlatformHandle() const { return ispcrtPlatformNativeHandle(handle()); }
inline void *Device::nativeDeviceHandle() const { return ispcrtDeviceNativeHandle(handle()); }
inline void *Device::nativeContextHandle() const { return ispcrtDeviceContextNativeHandle(handle()); }

inline ISPCRTDeviceType Device::getType() const { return ispcrtGetDeviceType(handle()); }

inline uint32_t Device::deviceCount(ISPCRTDeviceType type) { return ispcrtGetDeviceCount(type); }

inline ISPCRTDeviceInfo Device::deviceInformation(ISPCRTDeviceType type, uint32_t deviceIdx) {
    ISPCRTDeviceInfo devInfo;
    ispcrtGetDeviceInfo(type, deviceIdx, &devInfo);
    return devInfo;
}

inline std::vector<ISPCRTDeviceInfo> Device::allDevicesInformation(ISPCRTDeviceType type) {
    auto devCount = ispcrtGetDeviceCount(type);
    std::vector<ISPCRTDeviceInfo> devInfo(devCount);
    for (int i = 0; i < devCount; i++) {
        ispcrtGetDeviceInfo(type, i, &devInfo[i]);
    }
    return devInfo;
}

inline void Device::dynamicLinkModules(ISPCRTModule *modules, const uint32_t num) {
    ispcrtDynamicLinkModules(handle(), (ISPCRTModule *)modules, num);
}

inline ISPCRTAllocationType Device::getMemoryAllocType(void *memBuffer) {
    return ispcrtGetMemoryAllocType(handle(), memBuffer);
}

/////////////////////////////////////////////////////////////////////////////
// Arrays (MemoryView wrapper w/ element type) //////////////////////////////
/////////////////////////////////////////////////////////////////////////////

enum class AllocType { Device, Shared };

enum class SharedMemoryUsageHint {
    HostDeviceReadWrite,
    HostWriteDeviceRead,
    HostReadDeviceWrite,
    ApplicationManagedDevice
};

template <typename T, AllocType AT = AllocType::Device> class Array : public GenericObject<ISPCRTMemoryView> {
  public:
    template <AllocType alloc>
    using EnableForSharedAllocation = typename std::enable_if<(alloc == AllocType::Shared)>::type *;
    template <AllocType alloc>
    using EnableForDeviceAllocation = typename std::enable_if<(alloc == AllocType::Device)>::type *;

    Array() = default;

    //////// Constructors that can be used for Device memory allocations ////////

    // Construct from raw array //
    template <AllocType alloc = AT>
    Array(const Device &device, T *appMemory, size_t size, EnableForDeviceAllocation<alloc> = 0);

    // Construct from std:: containers (array + vector) //
    template <std::size_t N, AllocType alloc = AT>
    Array(const Device &device, std::array<T, N> &arr, EnableForDeviceAllocation<alloc> = 0);

    template <typename ALLOC_T, AllocType alloc = AT>
    Array(const Device &device, std::vector<T, ALLOC_T> &v, EnableForDeviceAllocation<alloc> = 0);

    // Construct from single object //
    template <AllocType alloc = AT> Array(const Device &device, T &obj, EnableForDeviceAllocation<alloc> = 0);

    //////// Constructors that can be used for Shared memory allocations ////////

    // Allocate single object in shared memory
    template <AllocType alloc = AT>
    Array(const Device &device, SharedMemoryUsageHint SMAT = SharedMemoryUsageHint::HostDeviceReadWrite,
          EnableForSharedAllocation<alloc> = 0);

    // Allocate single object in shared memory for context
    template <AllocType alloc = AT>
    Array(const Context &context, SharedMemoryUsageHint SMAT = SharedMemoryUsageHint::HostDeviceReadWrite,
          EnableForSharedAllocation<alloc> = 0);

    // Allocate multiple objects in shared memory
    template <AllocType alloc = AT>
    Array(const Device &device, size_t size, SharedMemoryUsageHint SMAT = SharedMemoryUsageHint::HostDeviceReadWrite,
          EnableForSharedAllocation<alloc> = 0);

    // Allocate multiple objects in shared memory for context
    template <AllocType alloc = AT>
    Array(const Context &context, size_t size, SharedMemoryUsageHint SMAT = SharedMemoryUsageHint::HostDeviceReadWrite,
          EnableForSharedAllocation<alloc> = 0);

    //////// Methods valid only for Device memory allocations ////////

    // For shared memory objects those will return the same pointer //
    template <AllocType alloc = AT> T *hostPtr(EnableForDeviceAllocation<alloc> = 0) const;
    template <AllocType alloc = AT> T *devicePtr(EnableForDeviceAllocation<alloc> = 0) const;

    //////// Methods valid for Shared memory allocations ////////

    template <AllocType alloc = AT> T *sharedPtr(EnableForSharedAllocation<alloc> = 0) const;

    template <AllocType alloc = AT> SharedMemoryUsageHint smType(EnableForSharedAllocation<alloc> = 0) const;

    //////// Methods for all types of memory allocations ////////

    size_t size() const;
    AllocType type() const;

  private:
    SharedMemoryUsageHint m_smuh{SharedMemoryUsageHint::HostDeviceReadWrite};
};

// Inlined definitions //

// Device memory allocations
template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Device &device, T *appMemory, size_t size, EnableForDeviceAllocation<alloc>)
    : GenericObject<ISPCRTMemoryView>() {
    ISPCRTNewMemoryViewFlags flags;
    flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
    flags.smHint = ISPCRT_SM_HOST_DEVICE_READ_WRITE;
    m_handle = ispcrtNewMemoryView(device.handle(), appMemory, size * sizeof(T), &flags);
}

template <typename T, AllocType AT>
template <std::size_t N, AllocType alloc>
inline Array<T, AT>::Array(const Device &device, std::array<T, N> &arr, EnableForDeviceAllocation<alloc>)
    : Array<T, AT>(device, arr.data(), N) {}

template <typename T, AllocType AT>
template <typename ALLOC_T, AllocType alloc>
inline Array<T, AT>::Array(const Device &device, std::vector<T, ALLOC_T> &v, EnableForDeviceAllocation<alloc>)
    : Array<T, AT>(device, v.data(), v.size()) {}

template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Device &device, T &obj, EnableForDeviceAllocation<alloc>)
    : Array<T, AT>(device, &obj, 1) {}

// Shared memory allocations
template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Device &device, SharedMemoryUsageHint smuh, EnableForSharedAllocation<alloc>)
    : Array<T, AT>(device, 1, smuh) {}

template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Context &context, SharedMemoryUsageHint smuh, EnableForSharedAllocation<alloc>)
    : Array<T, AT>(context, 1, smuh) {}

inline void set_shared_memory_view_flags(ISPCRTNewMemoryViewFlags *p, SharedMemoryUsageHint t) {
    p->allocType = ISPCRT_ALLOC_TYPE_SHARED;
    switch (t) {
    case SharedMemoryUsageHint::HostDeviceReadWrite:
        p->smHint = ISPCRT_SM_HOST_DEVICE_READ_WRITE;
        break;
    case SharedMemoryUsageHint::HostWriteDeviceRead:
        p->smHint = ISPCRT_SM_HOST_WRITE_DEVICE_READ;
        break;
    case SharedMemoryUsageHint::HostReadDeviceWrite:
        p->smHint = ISPCRT_SM_HOST_READ_DEVICE_WRITE;
        break;
    case SharedMemoryUsageHint::ApplicationManagedDevice:
        p->smHint = ISPCRT_SM_APPLICATION_MANAGED_DEVICE;
        break;
    default:
        throw std::bad_alloc();
    }
}

template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Device &device, size_t size, SharedMemoryUsageHint smuh,
                           EnableForSharedAllocation<alloc>)
    : m_smuh(smuh), GenericObject<ISPCRTMemoryView>() {
    ISPCRTNewMemoryViewFlags flags;
    set_shared_memory_view_flags(&flags, smuh);
    m_handle = ispcrtNewMemoryView(device.handle(), nullptr, size * sizeof(T), &flags);
}

template <typename T, AllocType AT>
template <AllocType alloc>
inline Array<T, AT>::Array(const Context &context, size_t size, SharedMemoryUsageHint smuh,
                           EnableForSharedAllocation<alloc>)
    : m_smuh(smuh), GenericObject<ISPCRTMemoryView>() {
    ISPCRTNewMemoryViewFlags flags;
    set_shared_memory_view_flags(&flags, smuh);
    m_handle = ispcrtNewMemoryViewForContext(context.handle(), nullptr, size * sizeof(T), &flags);
}

// Device-only methods

template <typename T, AllocType AT>
template <AllocType alloc>
inline T *Array<T, AT>::hostPtr(EnableForDeviceAllocation<alloc>) const {
    return (T *)ispcrtHostPtr(handle());
}

template <typename T, AllocType AT>
template <AllocType alloc>
inline T *Array<T, AT>::devicePtr(EnableForDeviceAllocation<alloc>) const {
    return (T *)ispcrtDevicePtr(handle());
}

// Shared-only methods

template <typename T, AllocType AT>
template <AllocType alloc>
inline T *Array<T, AT>::sharedPtr(EnableForSharedAllocation<alloc>) const {
    return (T *)ispcrtSharedPtr(handle());
}

template <typename T, AllocType AT>
template <AllocType alloc>
inline SharedMemoryUsageHint Array<T, AT>::smType(EnableForSharedAllocation<alloc>) const {
    return m_smuh;
}

// All other methods

template <typename T, AllocType AT> inline size_t Array<T, AT>::size() const {
    return ispcrtSize(handle()) / sizeof(T);
}

template <typename T, AllocType AT> inline AllocType Array<T, AT>::type() const { return AT; }

/////////////////////////////////////////////////////////////////////////////
// Shared Memory Allocator //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename T> class SharedMemoryAllocator {
  public:
    using value_type = T;

    SharedMemoryAllocator() = delete;
    SharedMemoryAllocator(const Context &context,
                          SharedMemoryUsageHint smuh = SharedMemoryUsageHint::HostDeviceReadWrite)
        : m_context(context), m_smuh(smuh) {}
    SharedMemoryAllocator(const SharedMemoryAllocator &) = default;
    ~SharedMemoryAllocator() = default;
    SharedMemoryAllocator &operator=(const SharedMemoryAllocator &) = delete;

    T *allocate(const size_t n);
    void deallocate(T *const p, const size_t n);

    SharedMemoryUsageHint smType() const { return m_smuh; }

  protected:
    Context m_context;
    SharedMemoryUsageHint m_smuh;
    std::unordered_map<void *, Array<T, AllocType::Shared>> m_ptrToArray;
};

// Inlined definitions //

template <typename T> inline T *SharedMemoryAllocator<T>::allocate(const size_t n) {
    // Allocate a memory that can be shared between the host and the device
    auto a = Array<T, AllocType::Shared>(m_context, n, m_smuh);

    void *ptr = a.sharedPtr();
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    m_ptrToArray[ptr] = a;

    return static_cast<T *>(ptr);
}

template <typename T> inline void SharedMemoryAllocator<T>::deallocate(T *const p, const size_t) {
    if (m_ptrToArray.find(p) == m_ptrToArray.end())
        throw std::invalid_argument("pointer not allocated with this allocator");
    m_ptrToArray.erase(p);
}

// Provide convinience type for shared memory allocations
template <typename T> using SharedVector = std::vector<T, ispcrt::SharedMemoryAllocator<T>>;

/////////////////////////////////////////////////////////////////////////////
// ModuleOptions wrapper ////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class ModuleOptions : public GenericObject<ISPCRTModuleOptions> {
  public:
    ModuleOptions(const Device &device);
    ModuleOptions(const Device &device, ISPCRTModuleType moduleType, bool libraryCompilation = false,
                  uint32_t stackSize = 0);
    ModuleOptions(ISPCRTModuleOptions);
    uint32_t stackSize();
    bool libraryCompilation();
    ISPCRTModuleType moduleType();
    void setStackSize(uint32_t);
    void setLibraryCompilation(bool);
    void setModuleType(ISPCRTModuleType);
};

// Inlined definitions //

inline ModuleOptions::ModuleOptions(const Device &device)
    : GenericObject<ISPCRTModuleOptions>(ispcrtNewModuleOptionsEmpty(device.handle())) {}

inline ModuleOptions::ModuleOptions(const Device &device, ISPCRTModuleType moduleType, bool libraryCompilation,
                                    uint32_t stackSize)
    : GenericObject<ISPCRTModuleOptions>(
          ispcrtNewModuleOptions(device.handle(), moduleType, libraryCompilation, stackSize)) {}

inline ModuleOptions::ModuleOptions(ISPCRTModuleOptions opts) : GenericObject<ISPCRTModuleOptions>(opts) {}

inline uint32_t ModuleOptions::stackSize() { return ispcrtModuleOptionsGetStackSize(handle()); }

inline bool ModuleOptions::libraryCompilation() { return ispcrtModuleOptionsGetLibraryCompilation(handle()); }

inline ISPCRTModuleType ModuleOptions::moduleType() { return ispcrtModuleOptionsGetModuleType(handle()); }

inline void ModuleOptions::setStackSize(uint32_t size) { return ispcrtModuleOptionsSetStackSize(handle(), size); }

inline void ModuleOptions::setLibraryCompilation(bool isLibraryCompilation) {
    return ispcrtModuleOptionsSetLibraryCompilation(handle(), isLibraryCompilation);
}

inline void ModuleOptions::setModuleType(ISPCRTModuleType type) {
    return ispcrtModuleOptionsSetModuleType(handle(), type);
}

/////////////////////////////////////////////////////////////////////////////
// Module wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Module : public GenericObject<ISPCRTModule> {
  public:
    Module() = default;
    Module(const Device &device, const char *moduleName);
    Module(const Device &device, const char *moduleName, const ModuleOptions &opts);
    Module(ISPCRTModule module);
    void *functionPtr(const char *functionName);
};

// Inlined definitions //

inline Module::Module(const Device &device, const char *moduleName)
    : GenericObject<ISPCRTModule>(ispcrtLoadModule(device.handle(), moduleName)) {}

inline Module::Module(const Device &device, const char *moduleName, const ModuleOptions &opts)
    : GenericObject<ISPCRTModule>(ispcrtLoadModuleWithOptions(device.handle(), moduleName, opts.handle())) {}

inline Module::Module(ISPCRTModule module) : GenericObject<ISPCRTModule>(module) {}

inline void *Module::functionPtr(const char *functionName) { return ispcrtFunctionPtr(handle(), functionName); }

inline Module Device::staticLinkModules(ISPCRTModule *modules, const uint32_t num) {
    return Module(ispcrtStaticLinkModules(handle(), (ISPCRTModule *)modules, num));
}

/////////////////////////////////////////////////////////////////////////////
// Kernel wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Kernel : public GenericObject<ISPCRTKernel> {
  public:
    Kernel() = default;
    Kernel(const Device &device, const Module &module, const char *kernelName);
};

// Inlined definitions //

inline Kernel::Kernel(const Device &device, const Module &module, const char *kernelName)
    : GenericObject<ISPCRTKernel>(ispcrtNewKernel(device.handle(), module.handle(), kernelName)) {}

/////////////////////////////////////////////////////////////////////////////
// CommandList wrapper //////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class CommandList : public GenericObject<ISPCRTCommandList> {
  public:
    CommandList() = default;
    CommandList(ISPCRTCommandList);

    void barrier();

    template <typename T, AllocType AT> Future copyToDevice(const Array<T, AT> &arr) const;
    template <typename T, AllocType AT> Future copyToHost(const Array<T, AT> &arr) const;
    template <typename T, AllocType AT>
    Future copyArray(const Array<T, AT> &arrDst, const Array<T, AT> &arrSrc, const size_t size) const;

    Future launch(const Kernel &k, size_t dim0) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const;

    template <typename T, AllocType AT> Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0) const;
    template <typename T, AllocType AT>
    Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1) const;
    template <typename T, AllocType AT>
    Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1, size_t dim2) const;

    void close();
    Fence submit();
    void reset();

    void enableTimestamps();
    void *nativeHandle() const;
};

inline CommandList::CommandList(ISPCRTCommandList l) : GenericObject<ISPCRTCommandList>(l) {
    if (l)
        ispcrtRetain(l);
}

inline void CommandList::barrier() { ispcrtCommandListBarrier(handle()); }

template <typename T, AllocType AT> inline Future CommandList::copyToDevice(const Array<T, AT> &arr) const {
    return ispcrtCommandListCopyToDevice(handle(), arr.handle());
}

template <typename T, AllocType AT> inline Future CommandList::copyToHost(const Array<T, AT> &arr) const {
    return ispcrtCommandListCopyToHost(handle(), arr.handle());
}

template <typename T, AllocType AT>
inline Future CommandList::copyArray(const Array<T, AT> &arrDst, const Array<T, AT> &arrSrc, const size_t size) const {
    return ispcrtCommandListCopyMemoryView(handle(), arrDst.handle(), arrSrc.handle(), size * sizeof(T));
}

inline Future CommandList::launch(const Kernel &k, size_t dim0) const {
    return ispcrtCommandListLaunch1D(handle(), k.handle(), nullptr, dim0);
}

inline Future CommandList::launch(const Kernel &k, size_t dim0, size_t dim1) const {
    return ispcrtCommandListLaunch2D(handle(), k.handle(), nullptr, dim0, dim1);
}

inline Future CommandList::launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtCommandListLaunch3D(handle(), k.handle(), nullptr, dim0, dim1, dim2);
}

template <typename T, AllocType AT>
inline Future CommandList::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0) const {
    return ispcrtCommandListLaunch1D(handle(), k.handle(), p.handle(), dim0);
}

template <typename T, AllocType AT>
inline Future CommandList::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1) const {
    return ispcrtCommandListLaunch2D(handle(), k.handle(), p.handle(), dim0, dim1);
}

template <typename T, AllocType AT>
inline Future CommandList::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtCommandListLaunch3D(handle(), k.handle(), p.handle(), dim0, dim1, dim2);
}

inline void CommandList::close() { ispcrtCommandListClose(handle()); }

inline Fence CommandList::submit() { return ispcrtCommandListSubmit(handle()); }

inline void CommandList::reset() { ispcrtCommandListReset(handle()); }

inline void CommandList::enableTimestamps() { ispcrtCommandListEnableTimestamps(handle()); }

inline void *CommandList::nativeHandle() const { return ispcrtCommandListNativeHandle(handle()); }

/////////////////////////////////////////////////////////////////////////////
// CommandQueue wrapper /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class CommandQueue : public GenericObject<ISPCRTCommandQueue> {
  public:
    CommandQueue() = default;
    CommandQueue(const Device &device, uint32_t ordinal);
    CommandList createCommandList();
    void sync();
    void *nativeHandle() const;
};

// Inlined definitions //

inline CommandQueue::CommandQueue(const Device &device, uint32_t ordinal)
    : GenericObject<ISPCRTCommandQueue>(ispcrtNewCommandQueue(device.handle(), ordinal)) {}

inline CommandList CommandQueue::createCommandList() { return ispcrtCommandQueueCreateCommandList(handle()); }

inline void CommandQueue::sync() { ispcrtCommandQueueSync(handle()); }

inline void *CommandQueue::nativeHandle() const { return ispcrtCommandQueueNativeHandle(handle()); }

/////////////////////////////////////////////////////////////////////////////
// TaskQueue wrapper ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class TaskQueue : public GenericObject<ISPCRTTaskQueue> {
  public:
    TaskQueue() = default;
    TaskQueue(const Device &device);

    void barrier() const;

    template <typename T, AllocType AT> void copyToDevice(const Array<T, AT> &arr) const;
    template <typename T, AllocType AT> void copyToHost(const Array<T, AT> &arr) const;
    template <typename T, AllocType AT>
    void copyArray(const Array<T, AT> &arrDst, const Array<T, AT> &arrSrc, const size_t size) const;

    Future launch(const Kernel &k, size_t dim0) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const;

    template <typename T, AllocType AT> Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0) const;
    template <typename T, AllocType AT>
    Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1) const;
    template <typename T, AllocType AT>
    Future launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1, size_t dim2) const;

    // wait for the command list to be executed (start the execution if needed as well)
    void sync() const;

    void *nativeTaskQueueHandle() const;
};

// Inlined definitions //

inline TaskQueue::TaskQueue(const Device &device)
    : GenericObject<ISPCRTTaskQueue>(ispcrtNewTaskQueue(device.handle())) {}

inline void TaskQueue::barrier() const { ispcrtDeviceBarrier(handle()); }

template <typename T, AllocType AT> inline void TaskQueue::copyToDevice(const Array<T, AT> &arr) const {
    ispcrtCopyToDevice(handle(), arr.handle());
}

template <typename T, AllocType AT> inline void TaskQueue::copyToHost(const Array<T, AT> &arr) const {
    ispcrtCopyToHost(handle(), arr.handle());
}

template <typename T, AllocType AT>
inline void TaskQueue::copyArray(const Array<T, AT> &arrDst, const Array<T, AT> &arrSrc, const size_t size) const {
    ispcrtCopyMemoryView(handle(), arrDst.handle(), arrSrc.handle(), size * sizeof(T));
}

inline Future TaskQueue::launch(const Kernel &k, size_t dim0) const {
    return ispcrtLaunch1D(handle(), k.handle(), nullptr, dim0);
}

inline Future TaskQueue::launch(const Kernel &k, size_t dim0, size_t dim1) const {
    return ispcrtLaunch2D(handle(), k.handle(), nullptr, dim0, dim1);
}

inline Future TaskQueue::launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtLaunch3D(handle(), k.handle(), nullptr, dim0, dim1, dim2);
}

template <typename T, AllocType AT>
inline Future TaskQueue::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0) const {
    return ispcrtLaunch1D(handle(), k.handle(), p.handle(), dim0);
}

template <typename T, AllocType AT>
inline Future TaskQueue::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1) const {
    return ispcrtLaunch2D(handle(), k.handle(), p.handle(), dim0, dim1);
}

template <typename T, AllocType AT>
inline Future TaskQueue::launch(const Kernel &k, const Array<T, AT> &p, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtLaunch3D(handle(), k.handle(), p.handle(), dim0, dim1, dim2);
}

inline void TaskQueue::sync() const { ispcrtSync(handle()); }

inline void *TaskQueue::nativeTaskQueueHandle() const { return ispcrtTaskQueueNativeHandle(handle()); }

} // namespace ispcrt
