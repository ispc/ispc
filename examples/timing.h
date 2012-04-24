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


#ifdef WIN32
#include <windows.h>
#define rdtsc __rdtsc
#else
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    __inline__ uint64_t rdtsc() {
        uint32_t low, high;
#ifdef __x86_64
        __asm__ __volatile__ (
            "xorl %%eax,%%eax \n    cpuid"
            ::: "%rax", "%rbx", "%rcx", "%rdx" );
#else
        __asm__ __volatile__ (
            "xorl %%eax,%%eax \n    cpuid"
            ::: "%eax", "%ebx", "%ecx", "%edx" );
#endif
        __asm__ __volatile__ (
                              "rdtsc" : "=a" (low), "=d" (high));
        return (uint64_t)high << 32 | low;
    }
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif            
            
static uint64_t start, end;

static inline void reset_and_start_timer()
{
    start = rdtsc();
}

/* Returns the number of millions of elapsed processor cycles since the
   last reset_and_start_timer() call. */
static inline double get_elapsed_mcycles()
{
    end = rdtsc();
    return (end-start) / (1024. * 1024.);
}
