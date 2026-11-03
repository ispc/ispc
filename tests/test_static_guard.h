/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Input arrays of the functional test harness, placed next to an inaccessible
// guard page, so that an access past the end (or before the start) of an
// array faults instead of reading or writing unrelated memory:
//
//   GuardTrailing:  [ ... | array ][ guard page ]   the array ends at the guard
//   GuardLeading:   [ guard page ][ array | ... ]   the array starts after it
//
// The size must be a multiple of 64 bytes, so that both placements keep the
// array 64-byte aligned. The memory is never freed: the test exits right
// after one run. WebAssembly has no page protection, so there the arrays are
// plain aligned allocations.

#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__WASM__)
#include <malloc.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

enum GuardSide { GuardLeading, GuardTrailing };

static inline void *AllocGuarded(size_t size, GuardSide side) {
    if (size == 0 || size % 64 != 0) {
        fprintf(stderr, "AllocGuarded: size %zu is not a positive multiple of 64\n", size);
        exit(1);
    }
#if defined(__WASM__)
    (void)side;
    void *p = memalign(64, size);
    if (p == NULL) {
        fprintf(stderr, "AllocGuarded: allocation of %zu bytes failed\n", size);
        exit(1);
    }
    return p;
#else
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    size_t page = info.dwPageSize;
#else
    long sc = sysconf(_SC_PAGESIZE);
    if (sc <= 0) {
        fprintf(stderr, "AllocGuarded: cannot get the page size\n");
        exit(1);
    }
    size_t page = (size_t)sc;
#endif
    size_t dataSize = (size + page - 1) / page * page;
    size_t total = dataSize + page;
#if defined(_WIN32) || defined(_WIN64)
    char *base = (char *)VirtualAlloc(NULL, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (base == NULL) {
        fprintf(stderr, "AllocGuarded: VirtualAlloc of %zu bytes failed (error %lu)\n", total, GetLastError());
        exit(1);
    }
    char *guard = side == GuardLeading ? base : base + dataSize;
    DWORD oldProtect;
    if (!VirtualProtect(guard, page, PAGE_NOACCESS, &oldProtect)) {
        fprintf(stderr, "AllocGuarded: VirtualProtect failed (error %lu)\n", GetLastError());
        exit(1);
    }
#else
    void *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) {
        perror("AllocGuarded: mmap");
        exit(1);
    }
    char *base = (char *)m;
    char *guard = side == GuardLeading ? base : base + dataSize;
    if (mprotect(guard, page, PROT_NONE) != 0) {
        perror("AllocGuarded: mprotect");
        exit(1);
    }
#endif
    return side == GuardLeading ? base + page : base + dataSize - size;
#endif // __WASM__
}
