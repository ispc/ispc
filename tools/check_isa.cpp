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
    static char isa_string[32];
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    return "ARM NEON";
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    enum ISA isa_id = get_x86_isa();
    if (isa_id == INVALID) {
        return "Unknown x86 ISA";
    }
    const char *isa = isa_strings[isa_id];
    // Show AMX status for AMX-capable ISAs (SPR, GNR, DMR - not NVL) and APX
    // status for APX-capable ISAs (NVL, DMR).
    if (isa_id == DMR_AVX10_2) {
        snprintf(isa_string, sizeof(isa_string), "%s (AMX %s, APX %s)", isa, __os_has_amx_support() ? "on" : "off",
                 get_x86_os_has_apx() ? "on" : "off");
    } else if (isa_id == SPR_AVX512 || isa_id == GNR_AVX512) {
        snprintf(isa_string, sizeof(isa_string), "%s (AMX %s)", isa, __os_has_amx_support() ? "on" : "off");
    } else if (isa_id == NVL_AVX10_2) {
        snprintf(isa_string, sizeof(isa_string), "%s (APX %s)", isa, get_x86_os_has_apx() ? "on" : "off");
    } else {
        return isa;
    }
    return isa_string;
#elif defined(__riscv)
    return "RISC-V";
#elif defined(__powerpc64__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    return "PPC64LE";
#else
#error "Unsupported host CPU architecture."
#endif
}

int main() {
    const char *isa = lGetSystemISA();
    printf("ISA: %s\n", isa);

    return 0;
}
