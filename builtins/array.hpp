/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file array.hpp
    @brief minimal implementation of std::array

    This header is added to avoid additional dependency on C++ library
    for builtins-c-cpu.cpp file.
*/

namespace notstd {

template <typename T, int sizeImpl> struct array {
    T dataImpl[sizeImpl];

    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using size_type = int;

    size_type size() const noexcept { return sizeImpl; }

    pointer data() noexcept { return &dataImpl[0]; }

    const_pointer data() const noexcept { return &dataImpl[0]; }

    iterator begin() noexcept { return data(); }

    const_iterator begin() const noexcept { return data(); }

    iterator end() noexcept { return data() + size(); }

    const_iterator end() const noexcept { return data() + size(); }

    const_reference operator[](int idx) const noexcept { return dataImpl[idx]; }

    reference operator[](int idx) noexcept { return dataImpl[idx]; }
};

} // namespace notstd
