/*
  Copyright (c) 2013-2020, Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This file is a standalone program, which detects the best supported ISA.  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>

#if defined(_WIN32) || defined(_WIN64)
#define HOST_IS_WINDOWS
#include <intrin.h>
#elif defined(__APPLE__)
#define HOST_IS_APPLE
#endif

#if !defined(__arm__) && !defined(__aarch64__)
#if !defined(HOST_IS_WINDOWS)
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(infoType));
}

/* Save %ebx in case it's the PIC register */
static void __cpuidex(int info[4], int level, int count) {
    __asm__ __volatile__("xchg{l}\t{%%}ebx, %1\n\t"
                         "cpuid\n\t"
                         "xchg{l}\t{%%}ebx, %1\n\t"
                         : "=a"(info[0]), "=r"(info[1]), "=c"(info[2]), "=d"(info[3])
                         : "0"(level), "2"(count));
}
#endif // !HOST_IS_WINDOWS

static bool __os_has_avx_support() {
#if defined(HOST_IS_WINDOWS)
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 6) == 6;
#else  // !defined(HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 6) == 6;
#endif // !defined(HOST_IS_WINDOWS)
}

static bool __os_has_avx512_support() {
#if defined(HOST_IS_WINDOWS)
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0xE6) == 0xE6;
#elif defined(HOST_IS_APPLE)
    // macOS has different way of dealing with AVX512 than Windows and Linux:
    // - by default AVX512 is off in the newly created thread, which means CPUID flags will
    //   indicate AVX512 availability, but OS support check (XCR0) will not succeed.
    // - AVX512 can be enabled either by calling thread_set_state() or by executing any
    //   AVX512 instruction, which would cause #UD exception handled by the OS.
    // The purpose of this check is to identify if AVX512 is potentially available, so we
    // need to bypass OS check and look at CPUID flags only.
    // See ispc issue #1854 for more details.
    return true;
#else  // !defined(HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0xE6) == 0xE6;
#endif // !defined(HOST_IS_WINDOWS)
}
#endif // !__arm__

static const char *lGetSystemISA() {
#if defined(__arm__) || defined(__aarch64__)
    return "ARM NEON";
#else
    int info[4];
    __cpuid(info, 1);

    int info2[4];
    // Call cpuid with eax=7, ecx=0
    __cpuidex(info2, 7, 0);

    if ((info[2] & (1 << 27)) != 0 &&  // OSXSAVE
        (info2[1] & (1 << 5)) != 0 &&  // AVX2
        (info2[1] & (1 << 16)) != 0 && // AVX512 F
        __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        if ((info2[1] & (1 << 17)) != 0 && // AVX512 DQ
            (info2[1] & (1 << 28)) != 0 && // AVX512 CDI
            (info2[1] & (1 << 30)) != 0 && // AVX512 BW
            (info2[1] & (1 << 31)) != 0) { // AVX512 VL
            return "SKX";
        } else if ((info2[1] & (1 << 26)) != 0 && // AVX512 PF
                   (info2[1] & (1 << 27)) != 0 && // AVX512 ER
                   (info2[1] & (1 << 28)) != 0) { // AVX512 CDI
            return "KNL";
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if ((info[2] & (1 << 27)) != 0 &&                           // OSXSAVE
        (info[2] & (1 << 28)) != 0 && __os_has_avx_support()) { // AVX
        // AVX1 for sure....
        // Ivy Bridge?
        if ((info[2] & (1 << 29)) != 0 && // F16C
            (info[2] & (1 << 30)) != 0) { // RDRAND
            // So far, so good.  AVX2?
            if ((info2[1] & (1 << 5)) != 0) {
                return "AVX2 (codename Haswell)";
            } else {
                // Ivy Bridge specific target was depricated in ISPC, but
                // no harm detecting it in standalone tool.
                return "AVX1.1 (codename Ivy Bridge)";
            }
        }
        // Regular AVX
        return "AVX (codename Sandy Bridge)";
    } else if ((info[2] & (1 << 19)) != 0) {
        return "SSE4";
    } else if ((info[3] & (1 << 26)) != 0) {
        return "SSE2";
    } else {
        return "Error";
    }
#endif
}

int main() {
    const char *isa = lGetSystemISA();
    printf("ISA: %s\n", isa);

    return 0;
}
