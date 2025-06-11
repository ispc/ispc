// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>

#ifdef _WIN32
#include <intrin.h>
#else
#ifdef IS_X86_ARCH
#include <x86intrin.h>
#endif // IS_X86_ARCH
#include <stdlib.h>
#include <string.h>
#define __forceinline __attribute__((always_inline))
#define _aligned_malloc(x, y) aligned_alloc(y, x)
#define _aligned_free free
#define isnan std::isnan
#define ceil std::ceil
#endif

#include <benchmark/benchmark.h>
#include <cfloat>
#include <malloc.h>
// #include "benchmark_markers.h"

#define RESTRICT __restrict /* no alias hint */
#define FORCEINLINE __forceinline

/* Integer types */
typedef signed char schar;
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef int64_t int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef uint64_t uint64;

typedef double FReal;

#define KINDA_SMALL_NUMBER (1.e-4f)
#define SMALL_NUMBER (1.e-8f)

#define ARGS Arg(100)->Arg(1000)->Arg(10000)
#define ARGS_LARGE Arg(10000)
#define ARGS_SMALL Arg(3)
#define ARGS_POW2 Arg(128)->Arg(1024)->Arg(8192)

static bool bErrorReturn;

#define BENCHMARK_DEFINE(ClassName, TestName, TestNumber)                                                              \
    BENCHMARK_DEFINE_F(ClassName, TestName)(benchmark::State & st) {                                                   \
        for (auto _ : st)                                                                                              \
            Run(st, TestNumber);                                                                                       \
        if (!ResultsCorrect(st, TestNumber)) {                                                                         \
            bErrorReturn = true;                                                                                       \
            st.SkipWithError("Incorrect results");                                                                     \
        }                                                                                                              \
    }

#define BENCHMARK_CASE_INNER(ClassName, TestName, TestNumber, ArgSize)                                                 \
    BENCHMARK_DEFINE(ClassName, TestName, TestNumber)                                                                  \
    BENCHMARK_REGISTER_F(ClassName, TestName)->ArgSize;

#define BENCHMARK_CASE(ClassName, TestName, TestNumber) BENCHMARK_CASE_INNER(ClassName, TestName, TestNumber, ARGS)

#define BENCHMARK_CASE_LARGE(ClassName, TestName, TestNumber)                                                          \
    BENCHMARK_CASE_INNER(ClassName, TestName, TestNumber, ARGS_LARGE)

#define BENCHMARK_CASE_SMALL(ClassName, TestName, TestNumber)                                                          \
    BENCHMARK_CASE_INNER(ClassName, TestName, TestNumber, ARGS_SMALL)

#define BENCHMARK_CASE_POW2(ClassName, TestName, TestNumber)                                                           \
    BENCHMARK_CASE_INNER(ClassName, TestName, TestNumber, ARGS_POW2)

#define DO_CHECK DEBUG          // Optionally turn off correctness checking on tests that accumulate to the input
#define DO_OPTIONAL_TESTS DEBUG // Optional tests that aren't normally used or change number of tests between targets

static const int StructAlignment = 64;

class TestBase : public benchmark::Fixture {
  public:
    virtual void SetUp(::benchmark::State &state) = 0;
    virtual void TearDown(::benchmark::State &state) = 0;
    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) = 0;
    virtual bool ResultsCorrect(const ::benchmark::State &state, unsigned int TestNumber) = 0;
};
