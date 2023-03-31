/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif
#if defined(_WIN32) || defined(_WIN64)
#define WINDOWS
#endif

#include "../../common/timing.h"
#include <algorithm>
#include <stdio.h>

#include "perfbench_ispc.h"

#ifdef WINDOWS
#define CALLINGCONV /*__vectorcall*/
#else
#define CALLINGCONV
#endif

typedef void(CALLINGCONV FuncType)(float *, int, float *, float *);

struct PerfTest {
    FuncType *aFunc;
    const char *aName;
    FuncType *bFunc;
    const char *bName;
    const char *testName;
};

extern void CALLINGCONV xyzSumAOS(float *a, int count, float *zeros, float *result);
extern void CALLINGCONV xyzSumSOA(float *a, int count, float *zeros, float *result);

static void lInitData(float *ptr, int count) {
    for (int i = 0; i < count; ++i)
        ptr[i] = float(i) / (1024.f * 1024.f);
}

static PerfTest tests[] = {
    {xyzSumAOS, "serial", ispc::xyzSumAOS, "ispc", "AOS vector element sum (with coalescing)"},
    {xyzSumAOS, "serial", ispc::xyzSumAOSStdlib, "ispc", "AOS vector element sum (stdlib swizzle)"},
    {xyzSumAOS, "serial", ispc::xyzSumAOSNoCoalesce, "ispc", "AOS vector element sum (no coalescing)"},
    {xyzSumSOA, "serial", ispc::xyzSumSOA, "ispc", "SOA vector element sum"},
    {xyzSumSOA, "serial", (FuncType *)ispc::xyzSumVarying, "ispc", "Varying vector element sum"},
    {ispc::gathers, "gather", ispc::loads, "vector load", "Memory reads"},
    {ispc::scatters, "scatter", ispc::stores, "vector store", "Memory writes"},
};

int main() {
    int count = 3 * 64 * 1024;
    float *a = new float[count];
    float zeros[32] = {0};

    int nTests = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < nTests; ++i) {
        lInitData(a, count);
        reset_and_start_timer();
        float resultA[3] = {0, 0, 0};
        for (int j = 0; j < 100; ++j)
            tests[i].aFunc(a, count, zeros, resultA);
        double aTime = get_elapsed_mcycles();

        lInitData(a, count);
        reset_and_start_timer();
        float resultB[3] = {0, 0, 0};
        for (int j = 0; j < 100; ++j)
            tests[i].bFunc(a, count, zeros, resultB);
        double bTime = get_elapsed_mcycles();

        printf("%-40s: [%.2f] M cycles %s, [%.2f] M cycles %s (%.2fx speedup).\n", tests[i].testName, aTime,
               tests[i].aName, bTime, tests[i].bName, aTime / bTime);
#if 0
        printf("\t(%f %f %f) - (%f %f %f)\n", resultSerial[0], resultSerial[1],
               resultSerial[2], resultISPC[0], resultISPC[1], resultISPC[2]);
#endif
    }

    return 0;
}
