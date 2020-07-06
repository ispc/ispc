// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace ispcrt {

class RefCounted {
  public:
    RefCounted() = default;
    virtual ~RefCounted() = default;

    RefCounted(const RefCounted &) = delete;
    RefCounted(RefCounted &&) = delete;

    RefCounted &operator=(const RefCounted &) = delete;
    RefCounted &operator=(RefCounted &&) = delete;

    void refInc() const;
    void refDec() const;
    long long useCount() const;

  private:
    mutable std::atomic<long long> refCounter{1};
};

// Inlined definitions //

inline void RefCounted::refInc() const { refCounter++; }

inline void RefCounted::refDec() const {
    if ((--refCounter) == 0)
        delete this;
}

inline long long RefCounted::useCount() const { return refCounter.load(); }

/////////////////////////////////////////////////////////////////////////////
// Pointer to a RefCounted object ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename T> class IntrusivePtr {
    static_assert(std::is_base_of<RefCounted, T>::value, "IntrusivePtr<T> can only be used with objects derived "
                                                         "from RefCounted");

  public:
    T *ptr{nullptr};

    IntrusivePtr() = default;
    ~IntrusivePtr();

    IntrusivePtr(const IntrusivePtr &input);
    IntrusivePtr(IntrusivePtr &&input);

    template <typename O> IntrusivePtr(const IntrusivePtr<O> &input);
    IntrusivePtr(T *const input);

    IntrusivePtr &operator=(const IntrusivePtr &input);
    IntrusivePtr &operator=(IntrusivePtr &&input);
    IntrusivePtr &operator=(T *input);

    operator bool() const;

    const T &operator*() const;
    T &operator*();

    const T *operator->() const;
    T *operator->();
};

// Inlined definitions //

template <typename T> inline IntrusivePtr<T>::~IntrusivePtr() {
    if (ptr)
        ptr->refDec();
}

template <typename T> inline IntrusivePtr<T>::IntrusivePtr(const IntrusivePtr<T> &input) : ptr(input.ptr) {
    if (ptr)
        ptr->refInc();
}

template <typename T> inline IntrusivePtr<T>::IntrusivePtr(IntrusivePtr<T> &&input) : ptr(input.ptr) {
    input.ptr = nullptr;
}

template <typename T>
template <typename O>
inline IntrusivePtr<T>::IntrusivePtr(const IntrusivePtr<O> &input) : ptr(input.ptr) {
    if (ptr)
        ptr->refInc();
}

template <typename T> inline IntrusivePtr<T>::IntrusivePtr(T *const input) : ptr(input) {
    if (ptr)
        ptr->refInc();
}

template <typename T> inline IntrusivePtr<T> &IntrusivePtr<T>::operator=(const IntrusivePtr &input) {
    if (input.ptr)
        input.ptr->refInc();
    if (ptr)
        ptr->refDec();
    ptr = input.ptr;
    return *this;
}

template <typename T> inline IntrusivePtr<T> &IntrusivePtr<T>::operator=(IntrusivePtr &&input) {
    ptr = input.ptr;
    input.ptr = nullptr;
    return *this;
}

template <typename T> inline IntrusivePtr<T> &IntrusivePtr<T>::operator=(T *input) {
    if (input)
        input->refInc();
    if (ptr)
        ptr->refDec();
    ptr = input;
    return *this;
}

template <typename T> inline IntrusivePtr<T>::operator bool() const { return ptr != nullptr; }

template <typename T> inline const T &IntrusivePtr<T>::operator*() const { return *ptr; }

template <typename T> inline T &IntrusivePtr<T>::operator*() { return *ptr; }

template <typename T> inline const T *IntrusivePtr<T>::operator->() const { return ptr; }

template <typename T> inline T *IntrusivePtr<T>::operator->() { return ptr; }

// Inlined operators ////////////////////////////////////////////////////////

template <typename T> inline bool operator<(const IntrusivePtr<T> &a, const IntrusivePtr<T> &b) {
    return a.ptr < b.ptr;
}

template <typename T> bool operator==(const IntrusivePtr<T> &a, const IntrusivePtr<T> &b) { return a.ptr == b.ptr; }

template <typename T> bool operator!=(const IntrusivePtr<T> &a, const IntrusivePtr<T> &b) { return a.ptr != b.ptr; }

} // namespace ispcrt
