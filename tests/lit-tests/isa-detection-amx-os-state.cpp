// Check the CPU detection helpers in src/isa.h with mocked CPUID/XCR0 values:
// OS AMX state must not change the detected ISA, CPU AMX support or the
// dispatch decision, and NVL (no AMX) must not select SPR/GNR/DMR variants.
// dispatch() mirrors module.cpp::lEmitISACompatibilityTest().
// The AVX10 version leaf 0x24 must be queried only when leaf 0 reports it.

// RUN: %{cxx} -x c++ -std=c++17 -DISA_H_MOCK_CPUID -I%S/../../src %s -o %t.bin
// RUN: %t.bin | FileCheck %s

// REQUIRES: X86_64_HOST && !MACOS_HOST

// CHECK: SPR os_amx=1: isa=10 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=10 dispatch(SPR,ICL)=10
// CHECK-NEXT: SPR os_amx=0: isa=10 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=10 dispatch(SPR,ICL)=10
// CHECK-NEXT: GNR os_amx=1: isa=11 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=11 dispatch(SPR,ICL)=10
// CHECK-NEXT: GNR os_amx=0: isa=11 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=11 dispatch(SPR,ICL)=10
// CHECK-NEXT: DMR os_amx=1: isa=13 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=11 dispatch(SPR,ICL)=10
// CHECK-NEXT: DMR os_amx=0: isa=13 cpu_has_amx=1 dispatch(GNR,SPR,ICL)=11 dispatch(SPR,ICL)=10
// CHECK-NEXT: NVL os_amx=1: isa=12 cpu_has_amx=0 dispatch(GNR,SPR,ICL)=9 dispatch(SPR,ICL)=9
// CHECK-NEXT: NVL os_amx=0: isa=12 cpu_has_amx=0 dispatch(GNR,SPR,ICL)=9 dispatch(SPR,ICL)=9
// CHECK-NEXT: DMR max_leaf=0x18: isa=11 leaf24_queried=0
// CHECK-NEXT: DMR max_leaf=0x23: isa=11 leaf24_queried=0
// CHECK-NEXT: DMR max_leaf=0x24: isa=13 leaf24_queried=1

#include <stdio.h>
#include <string.h>

// Mocked CPUID leaves 0, 1, (7,0), (7,1), (0x24,0) and XCR0.
static int g_leaf1[4], g_leaf7_0[4], g_leaf7_1[4], g_leaf24[4];
static int g_xcr0;
static int g_maxLeaf;
static bool g_leaf24Queried;

static void __cpuid(int info[4], int infoType) {
    memset(info, 0, 4 * sizeof(int));
    if (infoType == 0) {
        info[0] = g_maxLeaf; // highest basic leaf
    } else if (infoType == 1) {
        memcpy(info, g_leaf1, sizeof(g_leaf1));
    }
}

static void __cpuidex(int info[4], int level, int count) {
    memset(info, 0, 4 * sizeof(int));
    if (level == 7 && count == 0) {
        memcpy(info, g_leaf7_0, sizeof(g_leaf7_0));
    } else if (level == 7 && count == 1) {
        memcpy(info, g_leaf7_1, sizeof(g_leaf7_1));
    } else if (level == 0x24 && count == 0) {
        g_leaf24Queried = true;
        memcpy(info, g_leaf24, sizeof(g_leaf24));
    }
}

static int xgetbv() { return g_xcr0; }

#include "isa.h"

enum CPU { CPU_SPR, CPU_GNR, CPU_DMR, CPU_NVL };

static void setCPU(CPU cpu, bool osAMX) {
    memset(g_leaf1, 0, sizeof(g_leaf1));
    memset(g_leaf7_0, 0, sizeof(g_leaf7_0));
    memset(g_leaf7_1, 0, sizeof(g_leaf7_1));
    memset(g_leaf24, 0, sizeof(g_leaf24));
    g_maxLeaf = 0x24;
    g_leaf24Queried = false;

    // Leaf 1: SSE2, SSE4.1/4.2, OSXSAVE, AVX, F16C, RDRAND. EAX (the processor
    // signature) stays 0, so max_level must come from leaf 0.
    g_leaf1[2] = (1 << 19) | (1 << 20) | (1 << 27) | (1 << 28) | (1 << 29) | (1 << 30);
    g_leaf1[3] = (1 << 26);

    // Leaf (7,0): max subleaf 1; ICL baseline: AVX2, AVX512 F/DQ/CD/BW/VL, VBMI2, VNNI, BITALG, VPOPCNTDQ.
    g_leaf7_0[0] = 1;
    g_leaf7_0[1] = (1 << 5) | (1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (int)(1u << 31);
    g_leaf7_0[2] = (1 << 6) | (1 << 11) | (1 << 12) | (1 << 14);
    // AVX512-FP16 (all four CPUs), plus AVX-VNNI and AVX512-BF16 in leaf (7,1).
    g_leaf7_0[3] = (1 << 23);
    g_leaf7_1[0] = (1 << 4) | (1 << 5);

    // AMX-BF16, AMX-TILE, AMX-INT8.
    const int amx = (1 << 22) | (1 << 24) | (1 << 25);
    // AMX-FP16 in (7,1).EAX, PREFETCHI in (7,1).EDX.
    const int gnr_eax = (1 << 21), gnr_edx = (1 << 14);
    // CMPCCXADD, AVX-IFMA in (7,1).EAX; AVX-VNNI-INT8, AVX-NE-CONVERT, AVX-VNNI-INT16, AVX10, APX in (7,1).EDX.
    const int client_eax = (1 << 7) | (1 << 23);
    const int client_edx = (1 << 4) | (1 << 5) | (1 << 10) | (1 << 19) | (1 << 21);

    switch (cpu) {
    case CPU_SPR:
        g_leaf7_0[3] |= amx;
        break;
    case CPU_GNR:
        g_leaf7_0[3] |= amx;
        g_leaf7_1[0] |= gnr_eax;
        g_leaf7_1[3] |= gnr_edx;
        break;
    case CPU_DMR:
        g_leaf7_0[3] |= amx;
        g_leaf7_1[0] |= gnr_eax | client_eax;
        // AMX-COMPLEX.
        g_leaf7_1[3] |= gnr_edx | client_edx | (1 << 8);
        g_leaf24[1] = 2; // AVX10 version 2
        break;
    case CPU_NVL:
        g_leaf7_1[0] |= client_eax;
        g_leaf7_1[3] |= client_edx;
        g_leaf24[1] = 2; // AVX10 version 2
        break;
    }

    // XCR0: x87/SSE/AVX and AVX-512 state; optionally AMX XTILECFG/XTILEDATA.
    g_xcr0 = 0xE7 | (osAMX ? 0x60000 : 0);
}

// Mirror of the dispatcher's walk: the highest compiled candidate with
// system_isa >= candidate, where SPR/GNR/DMR additionally require CPU AMX.
static int dispatch(const int *candidates, int n) {
    int isa = get_x86_isa();
    int cpuHasAMX = get_x86_cpu_has_amx();
    for (int i = 0; i < n; ++i) {
        int c = candidates[i];
        bool requiresAMX = c == SPR_AVX512 || c == GNR_AVX512 || c == DMR_AVX10_2;
        if (isa >= c && (!requiresAMX || cpuHasAMX)) {
            return c;
        }
    }
    return INVALID;
}

int main() {
    const char *names[] = {"SPR", "GNR", "DMR", "NVL"};
    const int gnrSprIcl[] = {GNR_AVX512, SPR_AVX512, ICL_AVX512};
    const int sprIcl[] = {SPR_AVX512, ICL_AVX512};
    for (int cpu = CPU_SPR; cpu <= CPU_NVL; ++cpu) {
        for (int osAMX = 1; osAMX >= 0; --osAMX) {
            setCPU((CPU)cpu, osAMX);
            printf("%s os_amx=%d: isa=%d cpu_has_amx=%d dispatch(GNR,SPR,ICL)=%d dispatch(SPR,ICL)=%d\n", names[cpu],
                   osAMX, get_x86_isa(), get_x86_cpu_has_amx(), dispatch(gnrSprIcl, 3), dispatch(sprIcl, 2));
        }
    }

    // Leaf 0x24 is not queried below max leaf 0x24, so DMR without the AVX10
    // version falls back to GNR. 0x18 (decimal 24) and 0x23 guard against a
    // decimal or off-by-one comparison.
    const int maxLeaves[] = {0x18, 0x23, 0x24};
    for (int maxLeaf : maxLeaves) {
        setCPU(CPU_DMR, true);
        g_maxLeaf = maxLeaf;
        int isa = get_x86_isa();
        printf("DMR max_leaf=0x%x: isa=%d leaf24_queried=%d\n", maxLeaf, isa, g_leaf24Queried);
    }
    return 0;
}
