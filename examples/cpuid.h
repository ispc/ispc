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

#ifndef ISPC_CPUID_H
#define ISPC_CPUID_H 1

#ifdef _MSC_VER
// Provides a __cpuid() function with same signature as below
#include <intrin.h>
#else
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__ ("cpuid"
                          : "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
                          : "0" (infoType));
}
#endif

inline bool CPUSupportsSSE2() {
    int info[4];
    __cpuid(info, 1);
    return (info[3] & (1 << 26));
}

inline bool CPUSupportsSSE4() {
    int info[4];
    __cpuid(info, 1);
    return (info[2] & (1 << 19));
}

inline bool CPUSupportsAVX() {
    int info[4];
    __cpuid(info, 1);
    return (info[2] & (1 << 28));
}

#endif // ISPC_CPUID_H
