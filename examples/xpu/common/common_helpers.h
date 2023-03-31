/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

namespace hostutil {
struct Timings {
    uint64_t kernel_ns, host_ns;
    Timings(uint64_t ker, uint64_t hst) : kernel_ns(ker), host_ns(hst) {}
    void print(int niter) {
        double thost = host_ns / 1000000.0f / niter;
        double tkern = kernel_ns / 1000000.0f / niter;

        printf("%-18s%.2lf msec\n", "kern time:", tkern);
        printf("%-18s%.2lf msec\n", "host time:", thost);
    }
};

template <class T, size_t N> struct alignas(4096) PageAlignedArray { T data[N]; };
} // namespace hostutil

#endif
