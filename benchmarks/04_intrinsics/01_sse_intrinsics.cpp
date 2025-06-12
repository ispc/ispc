// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#if _MSC_VER
#pragma warning(disable : 4305)
#endif

#include "01_sse_intrinsics_ispc.h"
#include "TestBase.h"

#define IS_NAN(x) ((x) != (x))

static uint32 IntAllSet = 0xFFFFFFFF;
static uint32 FloatSignMask = 0x80000000;

static bool FloatEqual(const float F1, const float F2, const float Tolerance = KINDA_SMALL_NUMBER) {
    if (fabsf(F1 - F2) > Tolerance) {
        return false;
    }

    return true;
}

class mm_add_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_add_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] + Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_add_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_ss, ISPC, 2);

class mm_add_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_add_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] + Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_add_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_ps, ISPC, 2);

class mm_sub_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] - Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_sub_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] - Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_ss, ISPC, 2);

class mm_sub_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] - Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_sub_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] - Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_ps, ISPC, 2);

class mm_mul_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] * Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_mul_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mul_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] * Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_mul_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_ss, ISPC, 2);

class mm_mul_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] * Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_mul_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mul_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] * Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_mul_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_ps, ISPC, 2);

class mm_div_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k * 0.1f;
            Source2[k] = 1.0f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] / Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_div_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_div_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!FloatEqual(Result[k], Source1[k] / Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_div_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_ss, ISPC, 2);

class mm_div_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k * 0.1f;
            Source2[k] = 1.0f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] / Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_div_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_div_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!FloatEqual(Result[k], Source1[k] / Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_div_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_ps, ISPC, 2);

#if defined(__clang__)
#pragma clang optimize off
#endif
static float Sqrtf(const float A) { return sqrtf(A); }

static void Sqrtf_CPP(float *Result, const float *Source1, const int Iterations) {
    for (int k = 0; k < Iterations; k++) {
        Result[k] = sqrtf(Source1[k]);
    }
}
#if defined(__clang__)
#pragma clang optimize on
#endif

class mm_sqrt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            Sqrtf_CPP(Result, Source1, Iterations);
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 R = _mm_sqrt_ss(S1);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sqrt_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!FloatEqual(Result[k], Sqrtf(Source1[k]))) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_sqrt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_ss, ISPC, 2);

class mm_sqrt_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            Sqrtf_CPP(Result, Source1, Iterations);
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 R = _mm_sqrt_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sqrt_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!FloatEqual(Result[k], Sqrtf(Source1[k]))) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_sqrt_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_ps, ISPC, 2);

class mm_rcp_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 1.0f / (Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 R = _mm_rcp_ss(S1);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_rcp_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            // Check approximately equal since rcp is an approximation
            if (!FloatEqual(Result[k], 1.0f / (Source1[k]), 1.0e-3f)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_rcp_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rcp_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rcp_ss, ISPC, 2);

class mm_rcp_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 1.0f / (Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 R = _mm_rcp_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_rcp_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            // Check approximately equal since rcp is an approximation
            if (!FloatEqual(Result[k], 1.0f / (Source1[k]), 1.0e-3f)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_rcp_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rcp_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rcp_ps, ISPC, 2);

class mm_rsqrt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 1.0f / sqrtf(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 R = _mm_rsqrt_ss(S1);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_rsqrt_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            // Check approximately equal since rsqrt is an approximation
            if (!FloatEqual(Result[k], 1.0f / sqrtf(Source1[k]), 1.0e-3f)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_rsqrt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rsqrt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rsqrt_ss, ISPC, 2);

class mm_rsqrt_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 1.0f / sqrtf(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 R = _mm_rsqrt_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_rsqrt_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            // Check approximately equal since rsqrt is an approximation
            if (!FloatEqual(Result[k], 1.0f / sqrtf(Source1[k]), 1.0e-3f)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_rsqrt_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rsqrt_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_rsqrt_ps, ISPC, 2);

class mm_min_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fminf(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_min_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fminf(Source1[k], Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_min_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_ss, ISPC, 2);

class mm_min_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fminf(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_min_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fminf(Source1[k], Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_min_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_ps, ISPC, 2);

class mm_max_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmaxf(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_max_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmaxf(Source1[k], Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_max_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_ss, ISPC, 2);

class mm_max_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmaxf(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_max_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmaxf(Source1[k], Source2[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_max_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_ps, ISPC, 2);

class mm_and_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int *S1 = reinterpret_cast<int *>(&Source1[k]);
                int *S2 = reinterpret_cast<int *>(&Source2[k]);
                int R = ((*S1) & (*S2));
                Result[k] = *reinterpret_cast<float *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_and_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_and_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int *S1 = reinterpret_cast<int *>(&Source1[k]);
            int *S2 = reinterpret_cast<int *>(&Source2[k]);
            int R = ((*S1) & (*S2));
            if (Result[k] != *reinterpret_cast<float *>(&R)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_and_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_ps, ISPC, 2);

class mm_andnot_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int *S1 = reinterpret_cast<int *>(&Source1[k]);
                int *S2 = reinterpret_cast<int *>(&Source2[k]);
                int R = ((~(*S1)) & (*S2));
                Result[k] = *reinterpret_cast<float *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_andnot_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_andnot_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int *S1 = reinterpret_cast<int *>(&Source1[k]);
            int *S2 = reinterpret_cast<int *>(&Source2[k]);
            int R = ((~(*S1)) & (*S2));
            if (Result[k] != *reinterpret_cast<float *>(&R)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_andnot_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_ps, ISPC, 2);

class mm_or_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int *S1 = reinterpret_cast<int *>(&Source1[k]);
                int *S2 = reinterpret_cast<int *>(&Source2[k]);
                int R = ((*S1) | (*S2));
                Result[k] = *reinterpret_cast<float *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_or_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_or_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int *S1 = reinterpret_cast<int *>(&Source1[k]);
            int *S2 = reinterpret_cast<int *>(&Source2[k]);
            int R = ((*S1) | (*S2));
            if (Result[k] != *reinterpret_cast<float *>(&R)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_or_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_ps, ISPC, 2);

class mm_xor_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int *S1 = reinterpret_cast<int *>(&Source1[k]);
                int *S2 = reinterpret_cast<int *>(&Source2[k]);
                int R = ((*S1) ^ (*S2));
                Result[k] = *reinterpret_cast<float *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_xor_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_xor_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int *S1 = reinterpret_cast<int *>(&Source1[k]);
            int *S2 = reinterpret_cast<int *>(&Source2[k]);
            int R = ((*S1) ^ (*S2));
            if (Result[k] != *reinterpret_cast<float *>(&R)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_xor_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_ps, ISPC, 2);

class mm_cmpeq_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpeq_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpeq_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] == Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_ss, ISPC, 2);

class mm_cmpeq_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpeq_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpeq_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] == Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_ps, ISPC, 2);

class mm_cmplt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmplt_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmplt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] < Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_ss, ISPC, 2);

class mm_cmplt_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmplt_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmplt_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] < Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_ps, ISPC, 2);

class mm_cmple_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] <= Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmple_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmple_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] <= Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmple_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_ss, ISPC, 2);

class mm_cmple_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] <= Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmple_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmple_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] <= Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmple_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_ps, ISPC, 2);

class mm_cmpgt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpgt_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpgt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] > Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_ss, ISPC, 2);

class mm_cmpgt_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpgt_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpgt_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] > Source2[k] && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_ps, ISPC, 2);

class mm_cmpnle_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] <= Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpnle_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnle_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] <= Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnle_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_ss, ISPC, 2);

class mm_cmpnle_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] <= Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpnle_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnle_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] <= Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnle_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_ps, ISPC, 2);

class mm_cmpngt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] > Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpngt_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpngt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] > Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpngt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_ss, ISPC, 2);

class mm_cmpngt_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] > Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpngt_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpngt_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] > Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpngt_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_ps, ISPC, 2);

class mm_cmpnge_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] >= Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpnge_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnge_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] >= Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnge_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_ss, ISPC, 2);

class mm_cmpnge_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] >= Source2[k]) ? *reinterpret_cast<float *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpnge_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnge_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] >= Source2[k]) && *reinterpret_cast<uint32 *>(&Result[k]) != IntAllSet) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnge_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_ps, ISPC, 2);

class mm_cmpord_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k + 0.5f;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                float Value;
                if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                    Value = (float)0xFFFFFFFF;
                } else {
                    Value = 0.0f;
                }

                Result[k] = Value;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpord_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpord_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            float Value;
            if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                Value = (float)0xFFFFFFFF;
            } else {
                Value = 0.0f;
            }

            if (Result[k] != Value) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpord_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_ss, ISPC, 2);

class mm_cmpord_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k + 0.5f;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                float Value;
                if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                    Value = (float)0xFFFFFFFF;
                } else {
                    Value = 0.0f;
                }

                Result[k] = Value;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpord_ps(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpord_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            float Value;
            if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                Value = (float)0xFFFFFFFF;
            } else {
                Value = 0.0f;
            }

            if (Result[k] != Value) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpord_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_ps, ISPC, 2);

class mm_cmpunord_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k + 0.5f;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<float *>(&IntAllSet) : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                const __m128 R = _mm_cmpunord_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpunord_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] !=
                ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<float *>(&IntAllSet) : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpunord_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_ss, ISPC, 2);

class mm_cmpunord_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k + 0.5f;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<float *>(&IntAllSet) : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_cmpunord_ps(S1, S2);
                _mm_store_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpunord_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] !=
                ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<float *>(&IntAllSet) : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpunord_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_ps, ISPC, 2);

class mm_comieq_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] == Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comieq_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comieq_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] == Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comieq_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comieq_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comieq_ss, ISPC, 2);

class mm_comilt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] < Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comilt_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comilt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] < Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comilt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comilt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comilt_ss, ISPC, 2);

class mm_comile_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] <= Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comile_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comile_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] <= Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comile_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comile_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comile_ss, ISPC, 2);

class mm_comigt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] > Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comigt_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comigt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] > Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comigt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comigt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comigt_ss, ISPC, 2);

class mm_comige_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] >= Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comige_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comige_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] >= Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comige_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comige_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comige_ss, ISPC, 2);

class mm_comineq_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (IS_NAN(Source1[k]) || IS_NAN(Source2[k]) || (Source1[k] != Source2[k])) ? 1 : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_comineq_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comineq_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((IS_NAN(Source1[k]) || IS_NAN(Source2[k]) || (Source1[k] != Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_comineq_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comineq_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comineq_ss, ISPC, 2);

class mm_ucomieq_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] == Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomieq_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomieq_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] == Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomieq_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomieq_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomieq_ss, ISPC, 2);

class mm_ucomilt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] < Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomilt_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomilt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] < Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomilt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomilt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomilt_ss, ISPC, 2);

class mm_ucomile_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] <= Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomile_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomile_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] <= Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomile_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomile_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomile_ss, ISPC, 2);

class mm_ucomigt_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] > Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomigt_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomigt_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] > Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomigt_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomigt_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomigt_ss, ISPC, 2);

class mm_ucomige_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] >= Source2[k])) ? 1 : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomige_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomige_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((!IS_NAN(Source1[k]) && !IS_NAN(Source2[k]) && (Source1[k] >= Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomige_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomige_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomige_ss, ISPC, 2);

class mm_ucomineq_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (IS_NAN(Source1[k]) || IS_NAN(Source2[k]) || (Source1[k] != Source2[k])) ? 1 : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 S2 = _mm_load_ss(&Source2[k]);
                Result[k] = _mm_ucomineq_ss(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomineq_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((IS_NAN(Source1[k]) || IS_NAN(Source2[k]) || (Source1[k] != Source2[k])) ? 1 : 0)) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomineq_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomineq_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomineq_ss, ISPC, 2);

class mm_cvt_ss2si : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)(k - Iterations / 2) * 0.1f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (int)nearbyintf(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const int R = _mm_cvt_ss2si(S1);
                Result[k] = R;
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvt_ss2si(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (int)nearbyintf(Source1[k])) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_cvt_ss2si, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvt_ss2si, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvt_ss2si, ISPC, 2);

class mm_cvtt_ss2si : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)(k - Iterations / 2) * 0.1f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (int)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const int R = _mm_cvtt_ss2si(S1);
                Result[k] = R;
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtt_ss2si(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (int)Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtt_ss2si, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtt_ss2si, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtt_ss2si, ISPC, 2);

class mm_cvt_si2ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k + 0.5f;
            Source2[k] = (Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (float)Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const __m128 R = _mm_cvt_si2ss(S1, Source2[k]);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvt_si2ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (float)Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    int *Source2;
};

BENCHMARK_CASE_POW2(mm_cvt_si2ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvt_si2ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvt_si2ss, ISPC, 2);

class mm_cvtss_f32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                const float R = _mm_cvtss_f32(S1);
                Result[k] = R;
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtss_f32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtss_f32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtss_f32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtss_f32, ISPC, 2);

class mm_shuffle_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 3] + Source2[k + 3];
                Result[k + 1] = Source1[k + 2] + Source2[k + 2];
                Result[k + 2] = Source1[k + 1] + Source2[k + 1];
                Result[k + 3] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 S = _mm_add_ps(S1, S2);
                const __m128 R = _mm_shuffle_ps(S, S, 0x1B);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_shuffle_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 3] + Source2[k + 3] && Result[k + 1] != Source1[k + 2] + Source2[k + 2] &&
                Result[k + 2] != Source1[k + 1] + Source2[k + 1] && Result[k + 3] != Source1[k] + Source2[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_shuffle_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_ps, ISPC, 2);

class mm_unpackhi_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 2] + Source2[k + 2];
                Result[k + 1] = Source1[k + 2] + Source2[k + 2];
                Result[k + 2] = Source1[k + 3] + Source2[k + 3];
                Result[k + 3] = Source1[k + 3] + Source2[k + 3];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 S = _mm_add_ps(S1, S2);
                const __m128 R = _mm_unpackhi_ps(S, S);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 2] + Source2[k + 2] && Result[k + 1] != Source1[k + 2] + Source2[k + 2] &&
                Result[k + 2] != Source1[k + 3] + Source2[k + 3] && Result[k + 3] != Source1[k + 3] + Source2[k + 3]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_ps, ISPC, 2);

class mm_unpacklo_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k] + Source2[k];
                Result[k + 2] = Source1[k + 1] + Source2[k + 1];
                Result[k + 3] = Source1[k + 1] + Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 S = _mm_add_ps(S1, S2);
                const __m128 R = _mm_unpacklo_ps(S, S);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k] + Source2[k] &&
                Result[k + 2] != Source1[k + 1] + Source2[k + 1] && Result[k + 3] != Source1[k + 1] + Source2[k + 1]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_ps, ISPC, 2);

class mm_movehl_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 2] + Source2[k + 2];
                Result[k + 1] = Source1[k + 3] + Source2[k + 3];
                Result[k + 2] = Source1[k + 2] + Source2[k + 2];
                Result[k + 3] = Source1[k + 3] + Source2[k + 3];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 S = _mm_add_ps(S1, S2);
                const __m128 R = _mm_movehl_ps(S, S);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_movehl_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 2] + Source2[k + 2] && Result[k + 1] != Source1[k + 3] + Source2[k + 3] &&
                Result[k + 2] != Source1[k + 2] + Source2[k + 2] && Result[k + 3] != Source1[k + 3] + Source2[k + 3]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_movehl_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movehl_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movehl_ps, ISPC, 2);

class mm_movelh_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (float)k;
            Source2[k] = (float)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k + 1] + Source2[k + 1];
                Result[k + 2] = Source1[k] + Source2[k];
                Result[k + 3] = Source1[k + 1] + Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 S = _mm_add_ps(S1, S2);
                const __m128 R = _mm_movelh_ps(S, S);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_movelh_ps(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k + 1] + Source2[k + 1] &&
                Result[k + 2] != Source1[k] + Source2[k] && Result[k + 3] != Source1[k + 1] + Source2[k + 1]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_movelh_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movelh_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movelh_ps, ISPC, 2);

class mm_movemask_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                int R = 0;
                R |= *reinterpret_cast<int *>(&Source1[k]) & FloatSignMask ? 1 : 0;
                R |= *reinterpret_cast<int *>(&Source1[k + 1]) & FloatSignMask ? 1 << 1 : 0;
                R |= *reinterpret_cast<int *>(&Source1[k + 2]) & FloatSignMask ? 1 << 2 : 0;
                R |= *reinterpret_cast<int *>(&Source1[k + 3]) & FloatSignMask ? 1 << 3 : 0;
                Result[k] = R;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                Result[k] = _mm_movemask_ps(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_movemask_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            int R = 0;
            R |= *reinterpret_cast<int *>(&Source1[k]) & FloatSignMask ? 1 : 0;
            R |= *reinterpret_cast<int *>(&Source1[k + 1]) & FloatSignMask ? 1 << 1 : 0;
            R |= *reinterpret_cast<int *>(&Source1[k + 2]) & FloatSignMask ? 1 << 2 : 0;
            R |= *reinterpret_cast<int *>(&Source1[k + 3]) & FloatSignMask ? 1 << 3 : 0;

            if (Result[k] != R) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    int *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_movemask_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_ps, ISPC, 2);

class mm_set_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = 0.0f;
                Result[k + 2] = 0.0f;
                Result[k + 3] = 0.0f;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_set_ss(Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0.0f && Result[k + 2] != 0.0f && Result[k + 3] != 0.0f) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_set_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ss, ISPC, 2);

class mm_set_ps1 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
                Result[k + 2] = Source1[k];
                Result[k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_set_ps1(Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_ps1(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source1[k] && Result[k + 2] != Source1[k] &&
                Result[k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_set_ps1, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ps1, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ps1, ISPC, 2);

class mm_set_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations * 4];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];
        Source3 = new float[Iterations];
        Source4 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[4 * k] = 0;
            Result[4 * k + 1] = 0;
            Result[4 * k + 2] = 0;
            Result[4 * k + 3] = 0;
            Source1[k] = (float)k - (Iterations / 2);
            Source2[k] = (float)k;
            Source3[k] = (float)-k;
            Source4[k] = (float)k + 0.5f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[4 * k] = Source4[k];
                Result[4 * k + 1] = Source3[k];
                Result[4 * k + 2] = Source2[k];
                Result[4 * k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_set_ps(Source1[k], Source2[k], Source3[k], Source4[k]);
                _mm_storeu_ps(&Result[4 * k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_ps(Result, Source1, Source2, Source3, Source4, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[4 * k] != Source4[k] && Result[4 * k + 1] != Source3[k] && Result[4 * k + 2] != Source2[k] &&
                Result[4 * k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
    float *Source3;
    float *Source4;
};

BENCHMARK_CASE_POW2(mm_set_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_ps, ISPC, 2);

class mm_setr_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations * 4];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];
        Source3 = new float[Iterations];
        Source4 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[4 * k] = 0;
            Result[4 * k + 1] = 0;
            Result[4 * k + 2] = 0;
            Result[4 * k + 3] = 0;
            Source1[k] = (float)k - (Iterations / 2);
            Source2[k] = (float)k;
            Source3[k] = (float)-k;
            Source4[k] = (float)k + 0.5f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[4 * k] = Source1[k];
                Result[4 * k + 1] = Source2[k];
                Result[4 * k + 2] = Source3[k];
                Result[4 * k + 3] = Source4[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_setr_ps(Source1[k], Source2[k], Source3[k], Source4[k]);
                _mm_storeu_ps(&Result[4 * k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setr_ps(Result, Source1, Source2, Source3, Source4, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[4 * k] != Source1[k] && Result[4 * k + 1] != Source2[k] && Result[4 * k + 2] != Source3[k] &&
                Result[4 * k + 3] != Source4[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
    float *Source3;
    float *Source4;
};

BENCHMARK_CASE_POW2(mm_setr_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_ps, ISPC, 2);

class mm_setzero_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 1.0f;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 0.0f;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_setzero_ps();
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setzero_ps(Result, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != 0.0f) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;
    }

  private:
    int Iterations;

    float *Result;
};

BENCHMARK_CASE_POW2(mm_setzero_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_ps, ISPC, 2);

class mm_load_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                _mm_store_ss(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_load_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ss, ISPC, 2);

class mm_load_ps1 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
                Result[k + 2] = Source1[k];
                Result[k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_load_ps1(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_ps1(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source1[k] && Result[k + 2] != Source1[k] &&
                Result[k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_load_ps1, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ps1, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ps1, ISPC, 2);

class mm_load_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_load_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_load_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_ps, ISPC, 2);

class mm_loadr_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 3];
                Result[k + 1] = Source1[k + 2];
                Result[k + 2] = Source1[k + 1];
                Result[k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadr_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadr_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 3] && Result[k + 1] != Source1[k + 2] && Result[k + 2] != Source1[k + 1] &&
                Result[k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_loadr_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadr_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadr_ps, ISPC, 2);

class mm_loadu_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadu_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_loadu_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_ps, ISPC, 2);

class mm_store_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_load_ss(&Source1[k]);
                _mm_store_ss(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_store_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ss, ISPC, 2);

class mm_store_ps1 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
                Result[k + 2] = Source1[k];
                Result[k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_store_ps1(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_ps1(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source1[k] && Result[k + 2] != Source1[k] &&
                Result[k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_store_ps1, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ps1, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ps1, ISPC, 2);

class mm_store_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_store_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_store_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_ps, ISPC, 2);

class mm_storer_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 3];
                Result[k + 1] = Source1[k + 2];
                Result[k + 2] = Source1[k + 1];
                Result[k + 3] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storer_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storer_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 3] && Result[k + 1] != Source1[k + 2] && Result[k + 2] != Source1[k + 1] &&
                Result[k + 3] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_storer_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storer_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storer_ps, ISPC, 2);

class mm_storeu_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storeu_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_storeu_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_ps, ISPC, 2);

class mm_prefetch : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                _mm_prefetch((const char *)&Source1[k + 64], _MM_HINT_NTA);
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
        case 2:
            for (int k = 0; k < Iterations; k += 4) {
                _mm_prefetch((const char *)&Source1[k + 64], _MM_HINT_T0);
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
        case 3:
            for (int k = 0; k < Iterations; k += 4) {
                _mm_prefetch((const char *)&Source1[k + 64], _MM_HINT_T1);
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
        case 4:
            for (int k = 0; k < Iterations; k += 4) {
                _mm_prefetch((const char *)&Source1[k + 64], _MM_HINT_T2);
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_storeu_ps(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 5:
            ispc::mm_prefetch_nt(Result, Source1, Iterations);
            break;
        case 6:
            ispc::mm_prefetch_t0(Result, Source1, Iterations);
            break;
        case 7:
            ispc::mm_prefetch_t1(Result, Source1, Iterations);
            break;
        case 8:
            ispc::mm_prefetch_t2(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_prefetch, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_prefetch, Intrinsic_NT, 1);
BENCHMARK_CASE_POW2(mm_prefetch, Intrinsic_T0, 2);
BENCHMARK_CASE_POW2(mm_prefetch, Intrinsic_T1, 3);
BENCHMARK_CASE_POW2(mm_prefetch, Intrinsic_T2, 4);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_prefetch, ISPC_NT, 5);
BENCHMARK_CASE_POW2(mm_prefetch, ISPC_T0, 6);
BENCHMARK_CASE_POW2(mm_prefetch, ISPC_T1, 7);
BENCHMARK_CASE_POW2(mm_prefetch, ISPC_T2, 8);

class mm_stream_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                _mm_stream_ps(&Result[k], S1);
            }
            _mm_sfence();
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_stream_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_stream_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_stream_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_stream_ps, ISPC, 2);

class mm_move_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new float[Iterations];
        Source2 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (float)k - (Iterations / 2);
            Source2[k] = (float)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source2[k];
                Result[k + 1] = Source1[k + 1];
                Result[k + 2] = Source1[k + 2];
                Result[k + 3] = Source1[k + 3];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128 S2 = _mm_loadu_ps(&Source2[k]);
                const __m128 R = _mm_move_ss(S1, S2);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_move_ss(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source2[k] && Result[k + 1] != Source1[k + 1] && Result[k + 2] != Source1[k + 2] &&
                Result[k + 3] != Source1[k + 3]) {
                return false;
            }
        }

        return true;
    }

    virtual void TearDown(::benchmark::State &state) {
        delete[] Result;
        Result = nullptr;

        delete[] Source1;
        Source1 = nullptr;

        delete[] Source2;
        Source2 = nullptr;
    }

  private:
    int Iterations;

    float *Result;
    float *Source1;
    float *Source2;
};

BENCHMARK_CASE_POW2(mm_move_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_ss, ISPC, 2);

// Main function
BENCHMARK_MAIN();
