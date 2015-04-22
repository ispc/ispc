/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

#include <stdint.h>

#ifdef __arm__
#include <sys/time.h>
// There's no easy way to get a hardware clock counter on ARM, so instead
// we'll pretend it's a 1GHz processor and then compute pretend cycles
// based on elapsed time from gettimeofday().
__inline__ uint64_t rdtsc() {
  static bool first = true;
  static struct timeval tv_start;
  if (first) {
    gettimeofday(&tv_start, NULL);
    first = false;
    return 0;
  }

  struct timeval tv;
  gettimeofday(&tv, NULL);
  tv.tv_sec -= tv_start.tv_sec;
  tv.tv_usec -= tv_start.tv_usec;
  return (1000000ull * tv.tv_sec + tv.tv_usec) * 1000ull;
}

#include <sys/time.h>
static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}

#else // __arm__

#ifdef WIN32
#include <windows.h>
#define rdtsc __rdtsc
#else // WIN32
__inline__ uint64_t rdtsc() {
  uint32_t low, high;
#ifdef __x86_64
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%rax", "%rbx", "%rcx", "%rdx" );
#else
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%eax", "%ebx", "%ecx", "%edx" );
#endif
  __asm__ __volatile__ ("rdtsc" : "=a" (low), "=d" (high));
  return (uint64_t)high << 32 | low;
}

#include <sys/time.h>
static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}

#endif // !WIN32
#endif // !__arm__            
            
static uint64_t start,  end;
static double  tstart, tend;

static inline void reset_and_start_timer()
{
    start = rdtsc();
#ifndef WIN32
    // Unused in Windows build, rtc() causing link errors
    tstart = rtc();
#endif
}

/* Returns the number of millions of elapsed processor cycles since the
   last reset_and_start_timer() call. */
static inline double get_elapsed_mcycles()
{
    end = rdtsc();
    return (end-start) / (1024. * 1024.);
}

#ifndef WIN32
// Unused in Windows build, rtc() causing link errors
static inline double get_elapsed_msec()
{
    tend = rtc();
    return (tend - tstart)*1e3;
}
#endif
