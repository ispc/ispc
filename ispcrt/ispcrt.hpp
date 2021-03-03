// Copyright 2020-2021 Intel Corporation
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

inline Future::Future(ISPCRTFuture f) : GenericObject<ISPCRTFuture>(f) { if (f) ispcrtRetain(f); }

inline bool Future::valid() const { return handle() && ispcrtFutureIsValid(handle()); }

inline uint64_t Future::time() const { return ispcrtFutureGetTimeNs(handle()); }

/////////////////////////////////////////////////////////////////////////////
// Device wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Device : public GenericObject<ISPCRTDevice> {
  public:
    Device() = default;
    Device(ISPCRTDeviceType type);
    // deviceIdx is an index of the device in the list of supported devices
    // The list of the supported devices can be obtained with:
    // - allDevicesInformation() call or
    // - deviceCount() call and a series of deviceInformation() calls
    Device(ISPCRTDeviceType type, uint32_t deviceIdx);
    void* nativePlatformHandle() const;
    void* nativeDeviceHandle() const;
    void* nativeContextHandle() const;
    // static methods to get information about available devices
    static uint32_t deviceCount(ISPCRTDeviceType type);
    static ISPCRTDeviceInfo deviceInformation(ISPCRTDeviceType type, uint32_t deviceIdx);
    static std::vector<ISPCRTDeviceInfo> allDevicesInformation(ISPCRTDeviceType type);
};

// Inlined definitions //
inline Device::Device(ISPCRTDeviceType type, uint32_t deviceIdx) :
    GenericObject<ISPCRTDevice>(ispcrtGetDevice(type, deviceIdx)) { }
inline Device::Device(ISPCRTDeviceType type) : Device(type, 0) {}

inline void* Device::nativePlatformHandle() const { return ispcrtPlatformNativeHandle(handle()); }
inline void* Device::nativeDeviceHandle() const { return ispcrtDeviceNativeHandle(handle()); }
inline void* Device::nativeContextHandle() const { return ispcrtContextNativeHandle(handle()); }

inline uint32_t Device::deviceCount(ISPCRTDeviceType type) {
    return ispcrtGetDeviceCount(type);
}

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

/////////////////////////////////////////////////////////////////////////////
// Arrays (MemoryView wrapper w/ element type) //////////////////////////////
/////////////////////////////////////////////////////////////////////////////

enum class AllocType { Device, Shared };

template <typename T, AllocType AT = AllocType::Device> class Array : public GenericObject<ISPCRTMemoryView> {
  public:
    template<AllocType alloc> using EnableForSharedAllocation = typename std::enable_if<(alloc == AllocType::Shared)>::type*;
    template<AllocType alloc> using EnableForDeviceAllocation = typename std::enable_if<(alloc == AllocType::Device)>::type*;

    Array() = default;

    //////// Constructors that can be used for Device memory allocations ////////

    // Construct from raw array //
    template<AllocType alloc = AT>
        Array(const Device &device, T *appMemory, size_t size, EnableForDeviceAllocation<alloc> = 0);

    // Construct from std:: containers (array + vector) //
    template<std::size_t N, AllocType alloc = AT>
        Array(const Device &device, std::array<T, N> &arr, EnableForDeviceAllocation<alloc> = 0);

    template<typename ALLOC_T, AllocType alloc = AT>
        Array(const Device &device, std::vector<T, ALLOC_T> &v, EnableForDeviceAllocation<alloc> = 0);

    // Construct from single object //
    template<AllocType alloc = AT>
        Array(const Device &device, T &obj, EnableForDeviceAllocation<alloc> = 0);

    //////// Constructors that can be used for Shared memory allocations ////////

    // Allocate single object in shared memory
    template<AllocType alloc = AT>
        Array(const Device &device, EnableForSharedAllocation<alloc> = 0);

    // Allocate multiple objects in shared memory
    template<AllocType alloc = AT>
        Array(const Device &device, size_t size, EnableForSharedAllocation<alloc> = 0);

    //////// Methods valid only for Device memory allocations ////////

    // For shared memory objects those will return the same pointer //
    template<AllocType alloc = AT>
        T *hostPtr(EnableForDeviceAllocation<alloc> = 0) const;
    template<AllocType alloc = AT>
        T *devicePtr(EnableForDeviceAllocation<alloc> = 0) const;

    //////// Methods valid for Shared memory allocations ////////

    template<AllocType alloc = AT>
        T *sharedPtr(EnableForSharedAllocation<alloc> = 0) const;

    //////// Methods for all types of memory allocations ////////

    size_t size() const;
};

// Inlined definitions //

// Device memory allocations
template<typename T, AllocType AT>
template<AllocType alloc>
    inline Array<T, AT>::Array(const Device &device, T *appMemory, size_t size, EnableForDeviceAllocation<alloc>)
        : GenericObject<ISPCRTMemoryView>() {
            ISPCRTNewMemoryViewFlags flags;
            flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
            setHandle(ispcrtNewMemoryView(device.handle(), appMemory, size * sizeof(T), &flags));
        }

template<typename T, AllocType AT>
template<std::size_t N, AllocType alloc>
    inline Array<T,AT>::Array(const Device &device, std::array<T, N> &arr, EnableForDeviceAllocation<alloc>)
        : Array<T,AT>(device, arr.data(), N) {}

template<typename T, AllocType AT>
template<typename ALLOC_T, AllocType alloc>
    inline Array<T,AT>::Array(const Device &device, std::vector<T, ALLOC_T> &v, EnableForDeviceAllocation<alloc>)
        : Array<T,AT>(device, v.data(), v.size()) {}

template<typename T, AllocType AT>
template<AllocType alloc>
    inline Array<T,AT>::Array(const Device &device, T &obj, EnableForDeviceAllocation<alloc>) : Array<T,AT>(device, &obj, 1) {}

// Shared memory allocations
template<typename T, AllocType AT>
template<AllocType alloc>
    inline Array<T, AT>::Array(const Device &device, EnableForSharedAllocation<alloc>) : Array<T, AT>(device, 1) {}

template<typename T, AllocType AT>
template<AllocType alloc>
    inline Array<T, AT>::Array(const Device &device, size_t size, EnableForSharedAllocation<alloc>) :
        GenericObject<ISPCRTMemoryView>() {
            ISPCRTNewMemoryViewFlags flags;
            flags.allocType = ISPCRT_ALLOC_TYPE_SHARED;
            setHandle(ispcrtNewMemoryView(device.handle(), nullptr, size * sizeof(T), &flags));
        }

// Device-only methods

template<typename T, AllocType AT>
template<AllocType alloc>
    inline T *Array<T,AT>::hostPtr(EnableForDeviceAllocation<alloc>) const { return (T *)ispcrtHostPtr(handle()); }

template<typename T, AllocType AT>
template<AllocType alloc>
    inline T *Array<T,AT>::devicePtr(EnableForDeviceAllocation<alloc>) const { return (T *)ispcrtDevicePtr(handle()); }

// Shared-only methods

template<typename T, AllocType AT>
template<AllocType alloc>
    inline T *Array<T,AT>::sharedPtr(EnableForSharedAllocation<alloc>) const { return (T *)ispcrtSharedPtr(handle()); }

// All other methods

template <typename T, AllocType AT>
    inline size_t Array<T,AT>::size() const { return ispcrtSize(handle()) / sizeof(T); }

/////////////////////////////////////////////////////////////////////////////
// Shared Memory Allocator //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename T> class SharedMemoryAllocator {
  public:
    using value_type = T;

    SharedMemoryAllocator() = delete;
    SharedMemoryAllocator(const Device &device) : m_device(device) {}
    SharedMemoryAllocator(const SharedMemoryAllocator&) = default;
    ~SharedMemoryAllocator() = default;
    SharedMemoryAllocator& operator=(const SharedMemoryAllocator&) = delete;

    T* allocate(const size_t n);
    void deallocate(T* const p, const size_t n);
protected:
    Device m_device;
    std::unordered_map<void*, Array<T, AllocType::Shared>> m_ptrToArray;
};

// Inlined definitions //

template <typename T>
inline T *SharedMemoryAllocator<T>::allocate(const size_t n) {
    // Allocate a memory that can be shared between the host and the device
    auto a = Array<T, AllocType::Shared>(m_device, n);

    void* ptr = a.sharedPtr();
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    m_ptrToArray[ptr] = a;

    return static_cast<T*>(ptr);
}

template <typename T>
inline void SharedMemoryAllocator<T>::deallocate(T* const p, const size_t) {
    if (m_ptrToArray.find(p) == m_ptrToArray.end())
        throw std::invalid_argument("pointer not allocated with this allocator");
    m_ptrToArray.erase(p);
}

// Provide convinience type for shared memory allocations
template <typename T> using SharedVector = std::vector<T, ispcrt::SharedMemoryAllocator<T>>;

/////////////////////////////////////////////////////////////////////////////
// Module wrapper ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class Module : public GenericObject<ISPCRTModule> {
  public:
    Module() = default;
    Module(const Device &device, const char *moduleName);
};

// Inlined definitions //

inline Module::Module(const Device &device, const char *moduleName)
    : GenericObject<ISPCRTModule>(ispcrtLoadModule(device.handle(), moduleName)) {}

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
// TaskQueue wrapper ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class TaskQueue : public GenericObject<ISPCRTTaskQueue> {
  public:
    TaskQueue() = default;
    TaskQueue(const Device &device);

    void barrier() const;

    template <typename T, AllocType AT> void copyToDevice(const Array<T,AT> &arr) const;
    template <typename T, AllocType AT> void copyToHost(const Array<T,AT> &arr) const;

    Future launch(const Kernel &k, size_t dim0) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const;

    template <typename T, AllocType AT> Future launch(const Kernel &k, const Array<T,AT> &p, size_t dim0) const;
    template <typename T, AllocType AT> Future launch(const Kernel &k, const Array<T,AT> &p, size_t dim0, size_t dim1) const;
    template <typename T, AllocType AT>
    Future launch(const Kernel &k, const Array<T,AT> &p, size_t dim0, size_t dim1, size_t dim2) const;

    // start executing, but don't wait for the completion
    void submit() const;

    // wait for the command list to be executed (start the execution if needed as well)
    void sync() const;

    void* nativeTaskQueueHandle() const;
};

// Inlined definitions //

inline TaskQueue::TaskQueue(const Device &device)
    : GenericObject<ISPCRTTaskQueue>(ispcrtNewTaskQueue(device.handle())) {}

inline void TaskQueue::barrier() const { ispcrtDeviceBarrier(handle()); }

template <typename T, AllocType AT> inline void TaskQueue::copyToDevice(const Array<T,AT> &arr) const {
    ispcrtCopyToDevice(handle(), arr.handle());
}

template <typename T, AllocType AT> inline void TaskQueue::copyToHost(const Array<T,AT> &arr) const {
    ispcrtCopyToHost(handle(), arr.handle());
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

template <typename T, AllocType AT> inline Future TaskQueue::launch(const Kernel &k, const Array<T,AT> &p, size_t dim0) const {
    return ispcrtLaunch1D(handle(), k.handle(), p.handle(), dim0);
}

template <typename T, AllocType AT>
inline Future TaskQueue::launch(const Kernel &k, const Array<T,AT> &p, size_t dim0, size_t dim1) const {
    return ispcrtLaunch2D(handle(), k.handle(), p.handle(), dim0, dim1);
}

template <typename T, AllocType AT>
inline Future TaskQueue::launch(const Kernel &k, const Array<T,AT> &p, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtLaunch3D(handle(), k.handle(), p.handle(), dim0, dim1, dim2);
}

inline void TaskQueue::submit() const { ispcrtSubmit(handle()); }

inline void TaskQueue::sync() const { ispcrtSync(handle()); }

inline void* TaskQueue::nativeTaskQueueHandle() const { return ispcrtTaskQueueNativeHandle(handle()); }

} // namespace ispcrt
