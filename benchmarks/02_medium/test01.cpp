#include <benchmark/benchmark.h>
#include <cmath>
#include <stdio.h>

#include "../common.h"
#include "test01_ispc.h"

WARM_UP_RUN();

const float eps = 0.00001f;
#define ARGS Arg(100)->Arg(1000)->Arg(10000)

using namespace ispc;

static void init(struct FVector *dst, struct FVector *src0, struct FVector *src1, int count) {
    for (int i = 0; i < count; i++) {
        src0[i].V[0] = (float)i;
        src0[i].V[1] = (float)(i + 1);
        src0[i].V[2] = (float)(i + 2);
        src1[i].V[0] = 2;
        src1[i].V[1] = 4;
        src1[i].V[2] = 8;
        dst[i].V[0] = 0;
        dst[i].V[1] = 0;
        dst[i].V[2] = 0;
    }
}

static void check(struct FVector *dst, struct FVector *src0, struct FVector *src1, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(dst[i].V[j] - src0[i].V[j] * src1[i].V[j]) > eps) {
                printf("Error i=%d\n", i);
                return;
            }
        }
    }
}

static void test01_1(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    FVector *src0 = new FVector[count];
    FVector *src1 = new FVector[count];
    FVector *dst = new FVector[count];
    init(dst, src0, src1, count);

    for (auto _ : state) {
        TestUniform1(dst, src0, src1, count);
    }

    check(dst, src0, src1, count);
    delete[] src0;
    delete[] src1;
    delete[] dst;
}
BENCHMARK(test01_1)->ARGS;

static void test01_2(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    FVector *src0 = new FVector[count];
    FVector *src1 = new FVector[count];
    FVector *dst = new FVector[count];
    init(dst, src0, src1, count);

    for (auto _ : state) {
        TestUniform2(dst, src0, src1, count);
    }

    check(dst, src0, src1, count);
    delete[] src0;
    delete[] src1;
    delete[] dst;
}
BENCHMARK(test01_2)->ARGS;

static void test01_3(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    FVector *src0 = new FVector[count];
    FVector *src1 = new FVector[count];
    FVector *dst = new FVector[count];
    init(dst, src0, src1, count);

    for (auto _ : state) {
        TestUniform3(dst, src0, src1, count);
    }

    check(dst, src0, src1, count);
    delete[] src0;
    delete[] src1;
    delete[] dst;
}
BENCHMARK(test01_3)->ARGS;

BENCHMARK_MAIN();
