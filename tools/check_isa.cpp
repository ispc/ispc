/*
  Copyright (c) 2013-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This file is a standalone program, which detects the best supported ISA.  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "isa.h"

#include <stdio.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif // !defined(_MSC_VER)

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
static bool __os_enabled_amx_support() {
#if defined(_MSC_VER)
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0x60000) == 0x60000;
#else  // !defined(_MSC_VER)
    // Check xgetbv; this uses a .byte sequence instead of the instruction
    // directly because older assemblers do not include support for xgetbv and
    // there is no easy way to conditionally compile based on the assembler used.
    int rEAX = 0, rEDX = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(rEAX), "=d"(rEDX) : "c"(0));
    return (rEAX & 0x60000) == 0x60000;
#endif // !defined(_MSC_VER)
}
#endif // !__x86_64__

const char *const isa_strings[] = {
    "SSE2",
    "SSE4.1",
    "SSE4.2",
    "AVX (codename Sandy Bridge)",
    "AVX1.1 (codename Ivy Bridge)",
    "AVX2 (codename Haswell)",
    "AVX2VNNI (codename Alder Lake)",
    "KNL",
    "SKX",
    "ICL",
    "SPR",
    "GNR",
    "NVL",
    "DMR",
};

static const char *lGetSystemISA() {
    static char amx_isa_string[32];
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    return "ARM NEON";
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    enum ISA isa_id = get_x86_isa();
    if (isa_id == INVALID) {
        return "Unknown x86 ISA";
    }
    const char *isa = isa_strings[isa_id];
    // Only show AMX status for AMX-capable ISAs (SPR, GNR, DMR - not NVL)
    if (isa_id == SPR_AVX512 || isa_id == GNR_AVX512 || isa_id == DMR_AVX10_2) {
        snprintf(amx_isa_string, sizeof(amx_isa_string), "%s (AMX %s)", isa,
                 __os_enabled_amx_support() ? "on" : "off");
        return amx_isa_string;
    }
    return isa;
#elif defined(__riscv)
    return "RISC-V";
#else
#error "Unsupported host CPU architecture."
#endif
}

int main() {
    const char *isa = lGetSystemISA();
    printf("ISA: %s\n", isa);

    return 0;
}
