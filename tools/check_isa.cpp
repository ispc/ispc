/*
  Copyright (c) 2013-2025, Intel Corporation

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
    "DMR",
};

static const char *spr_amx_on = "SPR (AMX on)";
static const char *spr_amx_off = "SPR (AMX off)";

static const char *lGetSystemISA() {
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    return "ARM NEON";
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    enum ISA isa_id = get_x86_isa();
    if (isa_id == INVALID) {
        return "Unknown x86 ISA";
    }
    const char *isa = isa_strings[isa_id];
    if (isa_id == SPR_AVX512) {
        if (__os_enabled_amx_support()) {
            return spr_amx_on;
        } else {
            return spr_amx_off;
        }
    }
    return isa;
#else
#error "Unsupported host CPU architecture."
#endif
}

int main() {
    const char *isa = lGetSystemISA();
    printf("ISA: %s\n", isa);

    return 0;
}
