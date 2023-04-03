/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdint.h>

#if defined(__arm__) || defined(__aarch64__)
#include <stddef.h>
#include <sys/time.h>
// There's no easy way to get a hardware clock counter on ARM, so instead
// we'll pretend it's a 1GHz processor and then compute pretend cycles
// based on elapsed time from gettimeofday().
__inline__ uint64_t rdtsc() {
    static bool first = true;
    static struct timeval tv_start;
    if (first) {
        gettimeofday(&tv_start, nullptr);
        first = false;
        return 0;
    }

    struct timeval tv;
    gettimeofday(&tv, nullptr);
    tv.tv_sec -= tv_start.tv_sec;
    tv.tv_usec -= tv_start.tv_usec;
    return (1000000ull * tv.tv_sec + tv.tv_usec) * 1000ull;
}

#include <sys/time.h>
static inline double rtc(void) {
    struct timeval Tvalue;
    double etime;
    struct timezone dummy;

    gettimeofday(&Tvalue, &dummy);
    etime = (double)Tvalue.tv_sec + 1.e-6 * ((double)Tvalue.tv_usec);
    return etime;
}

#else // __arm__ || __aarch64__

#ifdef WIN32
#include <windows.h>
#define rdtsc __rdtsc
#else // WIN32
__inline__ uint64_t rdtsc() {
    uint32_t low, high;
#ifdef __x86_64
    __asm__ __volatile__("xorl %%eax,%%eax \n    cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
#else
    __asm__ __volatile__("xorl %%eax,%%eax \n    cpuid" ::: "%eax", "%ebx", "%ecx", "%edx");
#endif
    __asm__ __volatile__("rdtsc" : "=a"(low), "=d"(high));
    return (uint64_t)high << 32 | low;
}

#include <sys/time.h>
static inline double rtc(void) {
    struct timeval Tvalue;
    double etime;
    struct timezone dummy;

    gettimeofday(&Tvalue, &dummy);
    etime = (double)Tvalue.tv_sec + 1.e-6 * ((double)Tvalue.tv_usec);
    return etime;
}

#endif // !WIN32
#endif // !__arm__ && !__aarch64__

static uint64_t ustart, uend;
static double tstart, tend;

static inline void reset_and_start_timer() {
    ustart = rdtsc();
#ifndef WIN32
    // Unused in Windows build, rtc() causing link errors
    tstart = rtc();
#endif
}

/* Returns the number of millions of elapsed processor cycles since the
   last reset_and_start_timer() call. */
static inline double get_elapsed_mcycles() {
    uend = rdtsc();
    return (uend - ustart) / (1024. * 1024.);
}

#ifndef WIN32
// Unused in Windows build, rtc() causing link errors
static inline double get_elapsed_msec() {
    tend = rtc();
    return (tend - tstart) * 1e3;
}
#endif
