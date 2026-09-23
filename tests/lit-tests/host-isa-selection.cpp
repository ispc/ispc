/*
    Copyright (c) 2026, Intel Corporation

    SPDX-License-Identifier: BSD-3-Clause
*/

// Verify AMX/APX host ISA selection without depending on the physical host CPU.

// RUN: %{cxx} -std=c++17 -I%S/../../src %s -o %t.bin
// RUN: %t.bin

// REQUIRES: X86_64_HOST && !MACOS_HOST

#include "isa.h"
#include <cassert>

int main() {
    const struct {
        ISA detected;
        bool amxUsable;
        bool apxUsable;
        bool allAPXDisabled;
        ISA expected;
    } cases[] = {
        {SPR_AVX512, false, false, false, ICL_AVX512},  {GNR_AVX512, false, false, false, ICL_AVX512},
        {SPR_AVX512, true, false, false, SPR_AVX512},   {GNR_AVX512, true, false, false, GNR_AVX512},
        {DMR_AVX10_2, false, false, false, ICL_AVX512}, {DMR_AVX10_2, false, true, false, ICL_AVX512},
        {DMR_AVX10_2, false, true, true, ICL_AVX512},   {DMR_AVX10_2, true, false, false, GNR_AVX512},
        {DMR_AVX10_2, true, false, true, DMR_AVX10_2},  {DMR_AVX10_2, true, true, false, DMR_AVX10_2},
        {NVL_AVX10_2, false, false, false, ICL_AVX512}, {NVL_AVX10_2, false, false, true, NVL_AVX10_2},
        {NVL_AVX10_2, false, true, false, NVL_AVX10_2}, {AVX2, false, false, false, AVX2},
    };
    for (const auto &test : cases) {
        assert(get_x86_host_isa(test.detected, test.amxUsable, test.apxUsable, test.allAPXDisabled) == test.expected);
    }
    return 0;
}
