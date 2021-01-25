// Copyright 2020-2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ispcrt.h"
// std
#include <array>
#include <cassert>
#include <iostream>
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
    void* nativePlatformHandle() const;
    void* nativeDeviceHandle() const;
    void* nativeContextHandle() const;
};

// Inlined definitions //

inline Device::Device(ISPCRTDeviceType type) : GenericObject<ISPCRTDevice>(ispcrtGetDevice(type)) {}
inline void* Device::nativePlatformHandle() const { return ispcrtPlatformNativeHandle(handle()); }
inline void* Device::nativeDeviceHandle() const { return ispcrtDeviceNativeHandle(handle()); }
inline void* Device::nativeContextHandle() const { return ispcrtContextNativeHandle(handle()); }

/////////////////////////////////////////////////////////////////////////////
// Arrays (MemoryView wrapper w/ element type) //////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename T> class Array : public GenericObject<ISPCRTMemoryView> {
  public:
    Array() = default;

    // Construct from raw array //

    Array(const Device &device, T *appMemory, size_t size);

    // Construct from std:: containers (array + vector) //

    template <std::size_t N> Array(const Device &device, std::array<T, N> &arr);

    template <typename ALLOC_T> Array(const Device &device, std::vector<T, ALLOC_T> &v);

    // Construct from single object //

    Array(const Device &device, T &obj);

    T *hostPtr() const;
    T *devicePtr() const;

    size_t size() const;
};

// Inlined definitions //

template <typename T>
inline Array<T>::Array(const Device &device, T *appMemory, size_t size)
    : GenericObject<ISPCRTMemoryView>(ispcrtNewMemoryView(device.handle(), appMemory, size * sizeof(T))) {}

template <typename T>
template <std::size_t N>
inline Array<T>::Array(const Device &device, std::array<T, N> &arr) : Array<T>(device, arr.data(), N) {}

template <typename T>
template <typename ALLOC_T>
inline Array<T>::Array(const Device &device, std::vector<T, ALLOC_T> &v) : Array<T>(device, v.data(), v.size()) {}

template <typename T> inline Array<T>::Array(const Device &device, T &obj) : Array<T>(device, &obj, 1) {}

template <typename T> inline T *Array<T>::hostPtr() const { return (T *)ispcrtHostPtr(handle()); }

template <typename T> inline T *Array<T>::devicePtr() const { return (T *)ispcrtDevicePtr(handle()); }

template <typename T> inline size_t Array<T>::size() const { return ispcrtSize(handle()) / sizeof(T); }

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

    template <typename T> void copyToDevice(const Array<T> &arr) const;
    template <typename T> void copyToHost(const Array<T> &arr) const;

    Future launch(const Kernel &k, size_t dim0) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1) const;
    Future launch(const Kernel &k, size_t dim0, size_t dim1, size_t dim2) const;

    template <typename T> Future launch(const Kernel &k, const Array<T> &p, size_t dim0) const;
    template <typename T> Future launch(const Kernel &k, const Array<T> &p, size_t dim0, size_t dim1) const;
    template <typename T>
    Future launch(const Kernel &k, const Array<T> &p, size_t dim0, size_t dim1, size_t dim2) const;

    void sync() const;

    void* nativeTaskQueueHandle() const;
};

// Inlined definitions //

inline TaskQueue::TaskQueue(const Device &device)
    : GenericObject<ISPCRTTaskQueue>(ispcrtNewTaskQueue(device.handle())) {}

inline void TaskQueue::barrier() const { ispcrtDeviceBarrier(handle()); }

template <typename T> inline void TaskQueue::copyToDevice(const Array<T> &arr) const {
    ispcrtCopyToDevice(handle(), arr.handle());
}

template <typename T> inline void TaskQueue::copyToHost(const Array<T> &arr) const {
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

template <typename T> inline Future TaskQueue::launch(const Kernel &k, const Array<T> &p, size_t dim0) const {
    return ispcrtLaunch1D(handle(), k.handle(), p.handle(), dim0);
}

template <typename T>
inline Future TaskQueue::launch(const Kernel &k, const Array<T> &p, size_t dim0, size_t dim1) const {
    return ispcrtLaunch2D(handle(), k.handle(), p.handle(), dim0, dim1);
}

template <typename T>
inline Future TaskQueue::launch(const Kernel &k, const Array<T> &p, size_t dim0, size_t dim1, size_t dim2) const {
    return ispcrtLaunch3D(handle(), k.handle(), p.handle(), dim0, dim1, dim2);
}

inline void TaskQueue::sync() const { ispcrtSync(handle()); }

inline void* TaskQueue::nativeTaskQueueHandle() const { return ispcrtTaskQueueNativeHandle(handle()); }

} // namespace ispcrt
