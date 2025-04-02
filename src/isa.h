/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef ISA_H
#define ISA_H

enum ISA {
    INVALID = -1,

    SSE2 = 0,
    SSE41 = 1,
    SSE42 = 2,
    AVX = 3,
    AVX11 = 4,
    AVX2 = 5,
    AVX2VNNI = 6,
    KNL_AVX512 = 7,
    SKX_AVX512 = 8,
    ICL_AVX512 = 9,
    SPR_AVX512 = 10,
    AVX10_2_512 = 11,

    COUNT
};

// Define UNUSED_ATTR macro based on language standard and compiler support
#if defined(__cplusplus) && __cplusplus >= 201703L
// C++17 or newer
#define UNUSED_ATTR [[maybe_unused]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
// C23 or newer
#define UNUSED_ATTR [[maybe_unused]]
#elif defined(__clang__) || defined(__GNUC__)
// Clang/GCC specific attribute
#define UNUSED_ATTR __attribute__((unused))
#else
// No attribute support - define to nothing
#define UNUSED_ATTR
#endif

#ifndef MACOS
// MACOS macro can be defined when we are compiling dispatch.c for macOS.
// In other cases, we need to define it manually if we are compiling for macOS.
#if defined(__APPLE__)
#define MACOS
#endif
#endif // !MACOS

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

#if defined(_MSC_VER)
// __cpuid() and __cpuidex() are defined on Windows in <intrin.h> for x86/x64.
#include <intrin.h>
#else
// On *nix they need to be defined manually through inline assembler.
static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(infoType));
}

static void __cpuidex(int info[4], int level, int count) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(level), "2"(count));
}
#endif // !defined(_MSC_VER)

static int xgetbv() {
#if defined(_MSC_VER)
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
#else
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX = 0, rEDX = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return rEAX;
#endif // !defined(_MSC_VER)
}

static int __os_has_avx_support() { return (xgetbv() & 6) == 6; }

static int __os_has_avx512_support() {
#if !defined(MACOS)
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    return (xgetbv() & 0xE6) == 0xE6;
#else  // !MACOS
    // macOS has different way of dealing with AVX512 than Windows and Linux:
    // - by default AVX512 is off in the newly created thread, which means CPUID flags will
    //   indicate AVX512 availability, but OS support check (XCR0) will not succeed.
    // - AVX512 can be enabled either by calling thread_set_state() or by executing any
    //   AVX512 instruction, which would cause #UD exception handled by the OS.
    // The purpose of this check is to identify if AVX512 is potentially available, so we
    // need to bypass OS check and look at CPUID flags only.
    // See ispc issue #1854 for more details.
    return 1;
#endif // !MACOS
}

// Return of the x86 ISA enumerant values that gives the most capable ISA that
// the current system can run.
static enum ISA get_x86_isa() {
    int info[4];
    __cpuid(info, 1);
    UNUSED_ATTR int max_level = info[0];

    // Call cpuid with eax=7, ecx=0
    int info2[4];
    __cpuidex(info2, 7, 0);

    int info3[4] = {0, 0, 0, 0};
    int max_subleaf = info2[0];
    // Call cpuid with eax=7, ecx=1
    if (max_subleaf >= 1) {
        __cpuidex(info3, 7, 1);
    }

    // clang-format off
    int sse2 =                (info[3] & (1 << 26))  != 0;
    int sse41 =               (info[2] & (1 << 19))  != 0;
    int sse42 =               (info[2] & (1 << 20))  != 0;
    int avx =                 (info[2] & (1 << 28))  != 0;
    int avx2 =                (info2[1] & (1 << 5))  != 0;
    int avx_vnni =            (info3[0] & (1 << 4))  != 0;
    int avx_f16c =            (info[2] & (1 << 29))  != 0;
    int avx_rdrand =          (info[2] & (1 << 30))  != 0;
    int osxsave =             (info[2] & (1 << 27))  != 0;
    int avx512_f =            (info2[1] & (1 << 16)) != 0;

    UNUSED_ATTR int sha512 =              (info3[0] & (1 << 0)) != 0;
    UNUSED_ATTR int sm3 =                 (info3[0] & (1 << 1)) != 0;
    UNUSED_ATTR int sm4 =                 (info3[0] & (1 << 2)) != 0;
    UNUSED_ATTR int cmpccxadd =           (info3[0] & (1 << 7)) != 0;
    UNUSED_ATTR int amxfp16 =             (info3[0] & (1 << 21)) != 0;
    UNUSED_ATTR int avxifma =             (info3[0] & (1 << 23)) != 0;
    UNUSED_ATTR int avxvnniint8 =         (info3[3] & (1 << 4)) != 0;
    UNUSED_ATTR int avxneconvert =        (info3[3] & (1 << 5)) != 0;
    UNUSED_ATTR int amxcomplex =          (info3[3] & (1 << 8)) != 0;
    UNUSED_ATTR int avxvnniint16 =        (info3[3] & (1 << 10)) != 0;
    UNUSED_ATTR int prefetchi =           (info3[3] & (1 << 14)) != 0;
    // APX feature includes egpr, push2pop2, ppx, ndd, ccmp, nf, cf, zu
    UNUSED_ATTR int apx =                 (info3[3] & (1 << 21)) != 0;

    // clang-format on

    // NOTE: the values returned below must be the same as the
    // corresponding enumerant values in Target::ISA.
    if (osxsave && avx2 && avx512_f && __os_has_avx512_support()) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        // clang-format off
        int avx512_dq =           (info2[1] & (1 << 17)) != 0;
        int avx512_pf =           (info2[1] & (1 << 26)) != 0;
        int avx512_er =           (info2[1] & (1 << 27)) != 0;
        int avx512_cd =           (info2[1] & (1 << 28)) != 0;
        int avx512_bw =           (info2[1] & (1 << 30)) != 0;
        int avx512_vl =           (info2[1] & (1 << 31)) != 0;
#if !defined(MACOS)
        int avx512_vbmi2 =        (info2[2] & (1 << 6))  != 0;
        int avx512_gfni =         (info2[2] & (1 << 8))  != 0;
        int avx512_vaes =         (info2[2] & (1 << 9))  != 0;
        int avx512_vpclmulqdq =   (info2[2] & (1 << 10)) != 0;
        int avx512_vnni =         (info2[2] & (1 << 11)) != 0;
        int avx512_bitalg =       (info2[2] & (1 << 12)) != 0;
        int avx512_vpopcntdq =    (info2[2] & (1 << 14)) != 0;
        int avx512_bf16 =         (info3[0] & (1 << 5))  != 0;
        int avx512_vp2intersect = (info2[3] & (1 << 8))  != 0;
        int avx512_amx_bf16 =     (info2[3] & (1 << 22)) != 0;
        int avx512_amx_tile =     (info2[3] & (1 << 24)) != 0;
        int avx512_amx_int8 =     (info2[3] & (1 << 25)) != 0;
        int avx512_fp16 =         (info2[3] & (1 << 23)) != 0;
#endif // !MACOS
       // clang-format on

        // Knights Landing:          KNL = F + PF + ER + CD
        // Skylake server:           SKX = F + DQ + CD + BW + VL
        // Cascade Lake server:      CLX = SKX + VNNI
        // Cooper Lake server:       CPX = CLX + BF16
        // Ice Lake client & server: ICL = CLX + VBMI2 + GFNI + VAES + VPCLMULQDQ + BITALG + VPOPCNTDQ
        // Tiger Lake:               TGL = ICL + VP2INTERSECT
        // Sapphire Rapids:          SPR = ICL + BF16 + AMX_BF16 + AMX_TILE + AMX_INT8 + AVX_VNNI + FP16
        // Granite Rapids:           GNR = SPR + AMX_FP16 + PREFETCHI
        int knl = avx512_pf && avx512_er && avx512_cd;
        int skx = avx512_dq && avx512_cd && avx512_bw && avx512_vl;
#if !defined(MACOS)
        int clx = skx && avx512_vnni;
        UNUSED_ATTR int cpx = clx && avx512_bf16;
        int icl =
            clx && avx512_vbmi2 && avx512_gfni && avx512_vaes && avx512_vpclmulqdq && avx512_bitalg && avx512_vpopcntdq;
        UNUSED_ATTR int tgl = icl && avx512_vp2intersect;
        int spr =
            icl && avx512_bf16 && avx512_amx_bf16 && avx512_amx_tile && avx512_amx_int8 && avx_vnni && avx512_fp16;
        UNUSED_ATTR int gnr = spr && amxfp16 && prefetchi;

        int avx10 = (info3[3] & (1 << 19)) != 0;

        if (avx10) {
            // clang-format off

            int info_avx10[4] = {0, 0, 0, 0};
            if (max_level >= 24) {
                __cpuidex(info_avx10, 0x24, 0);
            }
            int avx10_ver = info_avx10[1] & 0xFF;
            int avx10_2 = avx10_ver >= 2;
            // clang-format on

            // Diamond Rapids:         DMR = GNR + AVX10_2_512 + APX + ... (For the whole list see x86TargetParser.cpp)
            int dmr = gnr && avx10_2 && apx && cmpccxadd && avxneconvert && avxifma && avxvnniint8 &&
                      avxvnniint16 && amxcomplex && sha512 && sm3 && sm4;

            if (dmr) {
                return AVX10_2_512;
            }
        }
        if (spr) {
            return SPR_AVX512;
        } else if (icl) {
            return ICL_AVX512;
        }
#endif // !MACOS
        if (skx) {
            return SKX_AVX512;
        } else if (knl) {
            return KNL_AVX512;
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if (osxsave && avx && __os_has_avx_support()) {
        if (avx_vnni) {
            return AVX2VNNI;
        }
        if (avx_f16c && avx_rdrand) {
            if (avx2) {
                return AVX2;
            } else {
                return AVX11;
            }
        }
        // Regular AVX
        return AVX;
    } else if (sse42) {
        return SSE42;
    } else if (sse41) {
        return SSE41;
    } else if (sse2) {
        return SSE2;
    }

    return INVALID;
}

#else

// For non-x86 platforms, define a function with trivial implementation.
static enum ISA get_x86_isa() { return INVALID; }

#endif // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

#endif // ISA_H
