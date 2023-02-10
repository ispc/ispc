/*
  Copyright (c) 2013-2023, Intel Corporation
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

#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM64)
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

static bool __os_enabled_amx_support() {
#if defined(HOST_IS_WINDOWS)
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0x60000) == 0x60000;
#else  // !defined(HOST_IS_WINDOWS)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0x60000) == 0x60000;
#endif // !defined(HOST_IS_WINDOWS)
}
#endif // !__arm__

static const char *lGetSystemISA() {
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    return "ARM NEON";
#else
    int info[4];
    __cpuid(info, 1);

    int info2[4];
    // Call cpuid with eax=7, ecx=0
    __cpuidex(info2, 7, 0);

    int info3[4];
    // Call cpuid with eax=7, ecx=1
    __cpuidex(info3, 7, 1);

    // clang-format off
    bool sse2 =                (info[3] & (1 << 26))  != 0;
    bool sse4 =                (info[2] & (1 << 19))  != 0;
    bool avx_f16c =            (info[2] & (1 << 29))  != 0;
    bool avx_rdrand =          (info[2] & (1 << 30))  != 0;
    bool osxsave =             (info[2] & (1 << 27))  != 0;
    bool avx =                 (info[2] & (1 << 28))  != 0;
    bool avx2 =                (info2[1] & (1 << 5))  != 0;
    bool avx512_f =            (info2[1] & (1 << 16)) != 0;
    bool avx512_dq =           (info2[1] & (1 << 17)) != 0;
    bool avx512_pf =           (info2[1] & (1 << 26)) != 0;
    bool avx512_er =           (info2[1] & (1 << 27)) != 0;
    bool avx512_cd =           (info2[1] & (1 << 28)) != 0;
    bool avx512_bw =           (info2[1] & (1 << 30)) != 0;
    bool avx512_vl =           (info2[1] & (1 << 31)) != 0;
    bool avx512_vbmi2 =        (info2[2] & (1 << 6))  != 0;
    bool avx512_gfni =         (info2[2] & (1 << 8))  != 0;
    bool avx512_vaes =         (info2[2] & (1 << 9))  != 0;
    bool avx512_vpclmulqdq =   (info2[2] & (1 << 10)) != 0;
    bool avx512_vnni =         (info2[2] & (1 << 11)) != 0;
    bool avx512_bitalg =       (info2[2] & (1 << 12)) != 0;
    bool avx512_vpopcntdq =    (info2[2] & (1 << 14)) != 0;
    bool avx_vnni =            (info3[0] & (1 << 4))  != 0;
    bool avx512_bf16 =         (info3[0] & (1 << 5))  != 0;
    bool avx512_vp2intersect = (info2[3] & (1 << 8))  != 0;
    bool avx512_amx_bf16 =     (info2[3] & (1 << 22)) != 0;
    bool avx512_amx_tile =     (info2[3] & (1 << 24)) != 0;
    bool avx512_amx_int8 =     (info2[3] & (1 << 25)) != 0;
    bool avx512_fp16 =         (info2[3] & (1 << 23)) != 0;
    // clang-format on

    if (osxsave && avx2 && avx512_f && __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        // Knights Landing:          KNL = F + PF + ER + CD
        // Skylake server:           SKX = F + DQ + CD + BW + VL
        // Cascade Lake server:      CLX = SKX + VNNI
        // Cooper Lake server:       CPX = CLX + BF16
        // Ice Lake client & server: ICL = CLX + VBMI2 + GFNI + VAES + VPCLMULQDQ + BITALG + VPOPCNTDQ
        // Tiger Lake:               TGL = ICL + VP2INTERSECT
        // Sapphire Rapids:          SPR = ICL + BF16 + AMX_BF16 + AMX_TILE + AMX_INT8 + AVX_VNNI + FP16
        bool knl = avx512_pf && avx512_er && avx512_cd;
        bool skx = avx512_dq && avx512_cd && avx512_bw && avx512_vl;
        bool clx = skx && avx512_vnni;
        bool cpx = clx && avx512_bf16;
        bool icl =
            clx && avx512_vbmi2 && avx512_gfni && avx512_vaes && avx512_vpclmulqdq && avx512_bitalg && avx512_vpopcntdq;
        bool tgl = icl && avx512_vp2intersect;
        bool spr =
            icl && avx512_bf16 && avx512_amx_bf16 && avx512_amx_tile && avx512_amx_int8 && avx_vnni && avx512_fp16;
#pragma unused(cpx, tgl)
        if (spr) {
            if (__os_enabled_amx_support()) {
                return "SPR (AMX on)";
            } else {
                return "SPR (AMX off)";
            }
        } else if (skx) {
            return "SKX";
        } else if (knl) {
            return "KNL";
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if (osxsave && avx && __os_has_avx_support()) {
        // AVX1 for sure....
        // Ivy Bridge?
        if (avx_f16c && avx_rdrand) {
            // So far, so good.  AVX2?
            if (avx2) {
                return "AVX2 (codename Haswell)";
            } else {
                // Ivy Bridge specific target was deprecated in ISPC, but
                // no harm detecting it in standalone tool.
                return "AVX1.1 (codename Ivy Bridge)";
            }
        }
        // Regular AVX
        return "AVX (codename Sandy Bridge)";
    } else if (sse4) {
        return "SSE4";
    } else if (sse2) {
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
