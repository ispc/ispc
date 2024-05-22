/*
  Copyright (c) 2013-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// This is the source code of __get_system_isa() function in the following files:
// - dispatch.ll
// - dispatch-macos.ll
//
// Compile in the following way:
// - clang dispatch.c -O2 -emit-llvm -S -c -DREGULAR -o isa_dispatch.ll
// - clang dispatch.c -O2 -emit-llvm -S -c -DMACOS   -o isa_dispatch-macos.ll
//
// Use the definition of __get_system_isa() from .ll file to update dispatch*.ll files.
// Note that attributes and metadata need to be defined "in-place" instead of by number.
//
// MACOS version: the key difference is absence of OS support check for AVX512
// - see issue #1854 for more details. Also it does not support ISAs newer than
// SKX, as no such Macs exist.

// Require one of the macros to be defined to make sure that it's not
// misspelled on the command line.
#if !defined(REGULAR) && !defined(MACOS)
#error "Either REGULAR or MACOS macro need to defined"
#endif

#include <stdint.h>
#include <stdlib.h>

static void __cpuid(int info[4], int infoType) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(infoType));
}

static void __cpuidex(int info[4], int level, int count) {
    __asm__ __volatile__("cpuid" : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3]) : "0"(level), "2"(count));
}

static int __os_has_avx_support() {
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 6) == 6;
}

#if !defined(MACOS)
static int __os_has_avx512_support() {
    // Check if the OS saves the XMM, YMM and ZMM registers, i.e. it supports AVX2 and AVX512.
    // See section 2.1 of software.intel.com/sites/default/files/managed/0d/53/319433-022.pdf
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX, rEDX;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0xE6) == 0xE6;
}
#endif // !MACOS

// __get_system_isa should return a value corresponding to one of the
// Target::ISA enumerant values that gives the most capable ISA that the
// current system can run.
int32_t __get_system_isa() {
    int info[4];
    __cpuid(info, 1);

    // Call cpuid with eax=7, ecx=0
    int info2[4];
    __cpuidex(info2, 7, 0);

    int info3[4] = {0, 0, 0, 0};
    int max_subleaf = info2[0];
    // Call cpuid with eax=7, ecx=1
    if (max_subleaf >= 1)
        __cpuidex(info3, 7, 1);

    // clang-format off
    _Bool sse2 =                (info[3] & (1 << 26))  != 0;
    _Bool sse41 =               (info[2] & (1 << 19))  != 0;
    _Bool sse42 =               (info[2] & (1 << 20))  != 0;
    _Bool avx =                 (info[2] & (1 << 28))  != 0;
    _Bool avx2 =                (info2[1] & (1 << 5))  != 0;
    _Bool avx_vnni =            (info3[0] & (1 << 4))  != 0;
    _Bool avx_f16c =            (info[2] & (1 << 29))  != 0;
    _Bool avx_rdrand =          (info[2] & (1 << 30))  != 0;
    _Bool osxsave =             (info[2] & (1 << 27))  != 0;
    _Bool avx512_f =            (info2[1] & (1 << 16)) != 0;
    // clang-format on

    // NOTE: the values returned below must be the same as the
    // corresponding enumerant values in Target::ISA.
    if (osxsave && avx2 && avx512_f
#if !defined(MACOS)
        && __os_has_avx512_support()
#endif // !MACOS
    ) {
        // We need to verify that AVX2 is also available,
        // as well as AVX512, because our targets are supposed
        // to use both.

        // clang-format off
        _Bool avx512_dq =           (info2[1] & (1 << 17)) != 0;
        _Bool avx512_pf =           (info2[1] & (1 << 26)) != 0;
        _Bool avx512_er =           (info2[1] & (1 << 27)) != 0;
        _Bool avx512_cd =           (info2[1] & (1 << 28)) != 0;
        _Bool avx512_bw =           (info2[1] & (1 << 30)) != 0;
        _Bool avx512_vl =           (info2[1] & (1 << 31)) != 0;
#if !defined(MACOS)
        _Bool avx512_vbmi2 =        (info2[2] & (1 << 6))  != 0;
        _Bool avx512_gfni =         (info2[2] & (1 << 8))  != 0;
        _Bool avx512_vaes =         (info2[2] & (1 << 9))  != 0;
        _Bool avx512_vpclmulqdq =   (info2[2] & (1 << 10)) != 0;
        _Bool avx512_vnni =         (info2[2] & (1 << 11)) != 0;
        _Bool avx512_bitalg =       (info2[2] & (1 << 12)) != 0;
        _Bool avx512_vpopcntdq =    (info2[2] & (1 << 14)) != 0;
        _Bool avx512_bf16 =         (info3[0] & (1 << 5))  != 0;
        _Bool avx512_vp2intersect = (info2[3] & (1 << 8))  != 0;
        _Bool avx512_amx_bf16 =     (info2[3] & (1 << 22)) != 0;
        _Bool avx512_amx_tile =     (info2[3] & (1 << 24)) != 0;
        _Bool avx512_amx_int8 =     (info2[3] & (1 << 25)) != 0;
        _Bool avx512_fp16 =         (info2[3] & (1 << 23)) != 0;
#endif // !MACOS
       // clang-format on

        // Knights Landing:          KNL = F + PF + ER + CD
        // Skylake server:           SKX = F + DQ + CD + BW + VL
        // Cascade Lake server:      CLX = SKX + VNNI
        // Cooper Lake server:       CPX = CLX + BF16
        // Ice Lake client & server: ICL = CLX + VBMI2 + GFNI + VAES + VPCLMULQDQ + BITALG + VPOPCNTDQ
        // Tiger Lake:               TGL = ICL + VP2INTERSECT
        // Sapphire Rapids:          SPR = ICL + BF16 + AMX_BF16 + AMX_TILE + AMX_INT8 + AVX_VNNI + FP16
        _Bool knl = avx512_pf && avx512_er && avx512_cd;
        _Bool skx = avx512_dq && avx512_cd && avx512_bw && avx512_vl;
#if !defined(MACOS)
        _Bool clx = skx && avx512_vnni;
        _Bool cpx = clx && avx512_bf16;
        _Bool icl =
            clx && avx512_vbmi2 && avx512_gfni && avx512_vaes && avx512_vpclmulqdq && avx512_bitalg && avx512_vpopcntdq;
        _Bool tgl = icl && avx512_vp2intersect;
        _Bool spr =
            icl && avx512_bf16 && avx512_amx_bf16 && avx512_amx_tile && avx512_amx_int8 && avx_vnni && avx512_fp16;
        if (spr) {
            return 9; // SPR
        } else if (icl) {
            return 8; // ICL
        }
#endif // !MACOS
        if (skx) {
            return 7; // SKX
        } else if (knl) {
            return 6; // KNL
        }
        // If it's unknown AVX512 target, fall through and use AVX2
        // or whatever is available in the machine.
    }

    if (osxsave && avx && __os_has_avx_support()) {
        if (avx_vnni) {
            return 5; // ADL
        }
        if (avx_f16c && avx_rdrand && avx2) {
            return 4;
        }
        // Regular AVX
        return 3;
    } else if (sse42) {
        return 2; // SSE4.2
    } else if (sse41) {
        return 1; // SSE4.1
    } else if (sse2) {
        return 0; // SSE2
    } else {
        abort();
    }
}
