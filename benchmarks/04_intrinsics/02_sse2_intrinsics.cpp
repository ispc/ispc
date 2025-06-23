// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "02_sse2_intrinsics_ispc.h"
#include "TestBase.h"

#if defined _MSC_VER && !defined __clang__ && defined DEBUG
#pragma warning(disable : 4324) // Silence warning for padding variables
#elif __clang__
#pragma GCC diagnostic ignored "-Wunused-private-field" // Silence warning for padding variables
#endif

#define IS_NAN(x) ((x) != (x))

static uint64 IntAllSet = 0xFFFFFFFFFFFFFFFF;
static uint64 DoubleSignMask = 0x8000000000000000;

static bool DoubleEqual(const double F1, const double F2, const double Tolerance = KINDA_SMALL_NUMBER) {
    if (abs(F1 - F2) > Tolerance) {
        return false;
    }

    return true;
}

class mm_add_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_add_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_sd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_add_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_sd, ISPC, 2);

class mm_add_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_add_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_pd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_add_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_pd, ISPC, 2);

class mm_sub_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_sub_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_sd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_sd, ISPC, 2);

class mm_sub_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_sub_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_pd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_pd, ISPC, 2);

class mm_mul_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_mul_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mul_sd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_mul_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_sd, ISPC, 2);

class mm_mul_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_mul_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mul_pd(Result, Source1, Source2, Iterations);
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_mul_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_pd, ISPC, 2);

#if defined(__clang__)
#pragma clang optimize off
#endif
static double Sqrt(const double A) { return sqrt(A); }

static void Sqrt_CPP(double *Result, const double *Source1, const int Iterations) {
    for (int k = 0; k < Iterations; k++) {
        Result[k] = sqrt(Source1[k]);
    }
}
#if defined(__clang__)
#pragma clang optimize on
#endif

class mm_sqrt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            Sqrt_CPP(Result, Source1, Iterations);
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d R = _mm_sqrt_sd(S1, S1);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sqrt_sd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!DoubleEqual(Result[k], Sqrt(Source1[k]))) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_sqrt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_sd, ISPC, 2);

class mm_sqrt_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)(k + 1);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            Sqrt_CPP(Result, Source1, Iterations);
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d R = _mm_sqrt_pd(S1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sqrt_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!DoubleEqual(Result[k], Sqrt(Source1[k]))) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_sqrt_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sqrt_pd, ISPC, 2);

class mm_div_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_div_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_div_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] / Source2[k]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_div_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_sd, ISPC, 2);

class mm_div_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_div_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_div_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Source1[k] / Source2[k]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_div_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_div_pd, ISPC, 2);

class mm_min_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmin(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_min_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmin(Source1[k], Source2[k])) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_min_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_sd, ISPC, 2);

class mm_min_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmin(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_min_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmin(Source1[k], Source2[k])) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_min_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_pd, ISPC, 2);

class mm_max_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmax(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_max_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmax(Source1[k], Source2[k])) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_max_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_sd, ISPC, 2);

class mm_max_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = fmax(Source1[k], Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_max_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != fmax(Source1[k], Source2[k])) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_max_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_pd, ISPC, 2);

class mm_and_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
                int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
                int64 R = ((*S1) & (*S2));
                Result[k] = *reinterpret_cast<double *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_and_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_and_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
            int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
            int64 R = ((*S1) & (*S2));
            if (Result[k] != *reinterpret_cast<double *>(&R)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_and_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_pd, ISPC, 2);

class mm_andnot_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
                int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
                int64 R = ((~(*S1)) & (*S2));
                Result[k] = *reinterpret_cast<double *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_andnot_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_andnot_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
            int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
            int64 R = ((~(*S1)) & (*S2));
            if (Result[k] != *reinterpret_cast<double *>(&R)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_andnot_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_pd, ISPC, 2);

class mm_or_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
                int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
                int64 R = ((*S1) | (*S2));
                Result[k] = *reinterpret_cast<double *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_or_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_or_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
            int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
            int64 R = ((*S1) | (*S2));
            if (Result[k] != *reinterpret_cast<double *>(&R)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_or_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_pd, ISPC, 2);

class mm_xor_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
                int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
                int64 R = ((*S1) ^ (*S2));
                Result[k] = *reinterpret_cast<double *>(&R);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_xor_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_xor_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            int64 *S1 = reinterpret_cast<int64 *>(&Source1[k]);
            int64 *S2 = reinterpret_cast<int64 *>(&Source2[k]);
            int64 R = ((*S1) ^ (*S2));
            if (Result[k] != *reinterpret_cast<double *>(&R)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_xor_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_pd, ISPC, 2);

class mm_cmpeq_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpeq_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpeq_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] == Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_sd, ISPC, 2);

class mm_cmpeq_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpeq_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpeq_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] == Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpeq_pd, ISPC, 2);

class mm_cmplt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmplt_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmplt_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] < Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_sd, ISPC, 2);

class mm_cmplt_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmplt_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmplt_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] < Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmplt_pd, ISPC, 2);

class mm_cmple_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] <= Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmple_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmple_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] <= Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmple_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_sd, ISPC, 2);

class mm_cmple_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] <= Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmple_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmple_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] <= Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmple_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmple_pd, ISPC, 2);

class mm_cmpgt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpgt_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpgt_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] > Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_sd, ISPC, 2);

class mm_cmpgt_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpgt_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpgt_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Source1[k] > Source2[k] && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpgt_pd, ISPC, 2);

class mm_cmpnle_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] <= Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpnle_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnle_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] <= Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnle_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_sd, ISPC, 2);

class mm_cmpnle_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] <= Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpnle_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnle_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] <= Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnle_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnle_pd, ISPC, 2);

class mm_cmpngt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] > Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpngt_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpngt_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] > Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpngt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_sd, ISPC, 2);

class mm_cmpngt_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] > Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpngt_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpngt_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] > Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpngt_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpngt_pd, ISPC, 2);

class mm_cmpnge_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] >= Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpnge_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnge_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] >= Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnge_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_sd, ISPC, 2);

class mm_cmpnge_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = !(Source1[k] >= Source2[k]) ? *reinterpret_cast<double *>(&IntAllSet) : 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpnge_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpnge_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (!(Source1[k] >= Source2[k]) && *reinterpret_cast<uint64 *>(&Result[k]) != IntAllSet) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpnge_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpnge_pd, ISPC, 2);

class mm_cmpord_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k + 0.5f;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                double Value;
                if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                    Value = 0xFFFFFFFF;
                } else {
                    Value = 0.0f;
                }

                Result[k] = Value;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpord_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpord_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            double Value;
            if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                Value = 0xFFFFFFFF;
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpord_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_sd, ISPC, 2);

class mm_cmpord_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k + 0.5f;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                double Value;
                if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                    Value = 0xFFFFFFFF;
                } else {
                    Value = 0.0f;
                }

                Result[k] = Value;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpord_pd(S1, S2);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpord_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            double Value;
            if ((!IS_NAN(Source1[k])) && (!IS_NAN(Source2[k]))) {
                Value = 0xFFFFFFFF;
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpord_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpord_pd, ISPC, 2);

class mm_cmpunord_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k + 0.5f;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<double *>(&IntAllSet) : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                const __m128d R = _mm_cmpunord_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpunord_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] !=
                ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<double *>(&IntAllSet) : 0)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpunord_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_sd, ISPC, 2);

class mm_cmpunord_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k + 0.5f;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<double *>(&IntAllSet) : 0);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d R = _mm_cmpunord_pd(S1, S2);
                _mm_store_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cmpunord_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] !=
                ((IS_NAN(Source1[k]) || IS_NAN(Source2[k])) ? *reinterpret_cast<double *>(&IntAllSet) : 0)) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpunord_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cmpunord_pd, ISPC, 2);

class mm_comieq_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comieq_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comieq_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comieq_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comieq_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comieq_sd, ISPC, 2);

class mm_comilt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comilt_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comilt_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comilt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comilt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comilt_sd, ISPC, 2);

class mm_comile_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comile_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comile_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comile_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comile_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comile_sd, ISPC, 2);

class mm_comigt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comigt_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comigt_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comigt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comigt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comigt_sd, ISPC, 2);

class mm_comige_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comige_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comige_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comige_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comige_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comige_sd, ISPC, 2);

class mm_comineq_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_comineq_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_comineq_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_comineq_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comineq_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_comineq_sd, ISPC, 2);

class mm_ucomieq_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomieq_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomieq_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomieq_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomieq_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomieq_sd, ISPC, 2);

class mm_ucomilt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomilt_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomilt_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomilt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomilt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomilt_sd, ISPC, 2);

class mm_ucomile_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomile_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomile_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomile_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomile_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomile_sd, ISPC, 2);

class mm_ucomigt_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomigt_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomigt_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomigt_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomigt_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomigt_sd, ISPC, 2);

class mm_ucomige_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomige_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomige_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomige_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomige_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomige_sd, ISPC, 2);

class mm_ucomineq_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                const __m128d S2 = _mm_load_sd(&Source2[k]);
                Result[k] = _mm_ucomineq_sd(S1, S2);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_ucomineq_sd(Result, Source1, Source2, Iterations);
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
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_ucomineq_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomineq_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_ucomineq_sd, ISPC, 2);

class mm_cvtepi32_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (double)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128d R = _mm_cvtepi32_pd(S1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtepi32_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (double)Source1[k]) {
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

    double *Result;
    int *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtepi32_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtepi32_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtepi32_pd, ISPC, 2);

class mm_cvtpd_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (int)nearbyint(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128i R = _mm_cvtpd_epi32(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtpd_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (int)nearbyint(Source1[k])) {
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtpd_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtpd_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtpd_epi32, ISPC, 2);

class mm_cvttpd_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128i R = _mm_cvttpd_epi32(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvttpd_epi32(Result, Source1, Iterations);
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvttpd_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttpd_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttpd_epi32, ISPC, 2);

class mm_cvtepi32_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (float)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_load_si128((__m128i *)&Source1[k]);
                const __m128 R = _mm_cvtepi32_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtepi32_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (float)Source1[k]) {
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
    int *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtepi32_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtepi32_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtepi32_ps, ISPC, 2);

class mm_cvtps_epi32 : public TestBase {
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
                Result[k] = (int)nearbyint(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128i R = _mm_cvtps_epi32(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtps_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (int)nearbyint(Source1[k])) {
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

BENCHMARK_CASE_POW2(mm_cvtps_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtps_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtps_epi32, ISPC, 2);

class mm_cvttps_epi32 : public TestBase {
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128i R = _mm_cvttps_epi32(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvttps_epi32(Result, Source1, Iterations);
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

BENCHMARK_CASE_POW2(mm_cvttps_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttps_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttps_epi32, ISPC, 2);

class mm_cvtpd_ps : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (float)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128 R = _mm_cvtpd_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtpd_ps(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (float)Source1[k]) {
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtpd_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtpd_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtpd_ps, ISPC, 2);

class mm_cvtps_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (float)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (double)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128 S1 = _mm_loadu_ps(&Source1[k]);
                const __m128d R = _mm_cvtps_pd(S1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtps_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (double)Source1[k]) {
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

    double *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtps_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtps_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtps_pd, ISPC, 2);

class mm_cvtsd_ss : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new float[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (float)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (float)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128 S1 = _mm_setzero_ps();
                const __m128d S2 = _mm_load_sd(&Source1[k]);
                const __m128 R = _mm_cvtsd_ss(S1, S2);
                _mm_store_ss(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsd_ss(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (float)Source1[k]) {
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsd_ss, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_ss, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_ss, ISPC, 2);

class mm_cvtss_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new float[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (float)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (double)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_setzero_pd();
                const __m128 S2 = _mm_load_ss(&Source1[k]);
                const __m128d R = _mm_cvtss_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtss_sd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (double)Source1[k]) {
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

    double *Result;
    float *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtss_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtss_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtss_sd, ISPC, 2);

class mm_cvtsd_si32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (int)nearbyint(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                Result[k] = _mm_cvtsd_si32(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsd_si32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (int)nearbyint(Source1[k])) {
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsd_si32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_si32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_si32, ISPC, 2);

class mm_cvttsd_si32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                Result[k] = _mm_cvttsd_si32(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvttsd_si32(Result, Source1, Iterations);
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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvttsd_si32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttsd_si32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttsd_si32, ISPC, 2);

class mm_cvtsi32_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (double)Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k++) {
                const __m128d S1 = _mm_setzero_pd();
                const int S2 = Source1[k];
                const __m128d R = _mm_cvtsi32_sd(S1, S2);
                _mm_store_sd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi32_sd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (double)Source1[k]) {
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

    double *Result;
    int *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsi32_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi32_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi32_sd, ISPC, 2);

class mm_unpackhi_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1] + Source2[k + 1];
                Result[k + 1] = Source1[k + 1] + Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                const __m128d R = _mm_unpackhi_pd(S, S);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] + Source2[k + 1] && Result[k + 1] != Source1[k + 1] + Source2[k + 1]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_pd, ISPC, 2);

class mm_unpacklo_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                const __m128d R = _mm_unpacklo_pd(S, S);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k] + Source2[k]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_pd, ISPC, 2);

class mm_movemask_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                int R = 0;
                R |= *reinterpret_cast<int64 *>(&Source1[k]) & DoubleSignMask ? 1 : 0;
                R |= *reinterpret_cast<int64 *>(&Source1[k + 1]) & DoubleSignMask ? 1 << 1 : 0;
                Result[k] = R;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                Result[k] = _mm_movemask_pd(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_movemask_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            int R = 0;
            R |= *reinterpret_cast<int64 *>(&Source1[k]) & DoubleSignMask ? 1 : 0;
            R |= *reinterpret_cast<int64 *>(&Source1[k + 1]) & DoubleSignMask ? 1 << 1 : 0;

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
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_movemask_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_pd, ISPC, 2);

class mm_shuffle_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1] + Source2[k + 1];
                Result[k + 1] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                const __m128d R = _mm_shuffle_pd(S, S, 0x1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_shuffle_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] + Source2[k + 1] && Result[k + 1] != Source1[k] + Source2[k]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_shuffle_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_pd, ISPC, 2);

class mm_load_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load_pd(&Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_load_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_pd, ISPC, 2);

class mm_load1_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load1_pd(&Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load1_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_load1_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load1_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load1_pd, ISPC, 2);

class mm_loadr_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadr_pd(&Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadr_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] && Result[k + 1] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_loadr_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadr_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadr_pd, ISPC, 2);

class mm_loadu_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadu_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_loadu_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_pd, ISPC, 2);

class mm_load_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                _mm_store_sd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_sd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_load_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_sd, ISPC, 2);

class mm_loadh_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k] + Source1[k];
                Result[k + 1] = Source1[k + 1] + Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load1_pd(&Source1[k]);
                const __m128d S2 = _mm_loadh_pd(S1, &Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                _mm_storeu_pd(&Result[k], S);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadh_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] + Source1[k] && Result[k + 1] != Source1[k + 1] + Source2[k + 1]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_loadh_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadh_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadh_pd, ISPC, 2);

class mm_loadl_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k + 1] + Source1[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load1_pd(&Source1[k]);
                const __m128d S2 = _mm_loadl_pd(S1, &Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                _mm_storeu_pd(&Result[k], S);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadl_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k + 1] + Source2[k + 1]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_loadl_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadl_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadl_pd, ISPC, 2);

class mm_set_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = 0.0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_set_sd(Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_sd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k] != 0.0) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_set_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_sd, ISPC, 2);

class mm_set1_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_set1_pd(Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set1_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_set1_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_pd, ISPC, 2);

class mm_set_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_set_pd(Source1[k], Source1[k + 1]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] && Result[k + 1] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_set_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_pd, ISPC, 2);

class mm_setr_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_setr_pd(Source1[k], Source1[k + 1]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setr_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_setr_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_pd, ISPC, 2);

class mm_setzero_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 0.0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1: {
            const __m128d S1 = _mm_setzero_pd();
            for (int k = 0; k < Iterations; k += 2) {
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
        }
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setzero_pd(Result, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != 0.0) {
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

    double *Result;
};

BENCHMARK_CASE_POW2(mm_setzero_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_pd, ISPC, 2);

class mm_move_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                const __m128d R = _mm_move_sd(S1, S);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_move_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k + 1]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_move_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_sd, ISPC, 2);

class mm_store_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                _mm_store_sd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_sd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_store_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_sd, ISPC, 2);

class mm_store1_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load_sd(&Source1[k]);
                _mm_store1_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store1_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_store1_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store1_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store1_pd, ISPC, 2);

class mm_store_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                _mm_store_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_store_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_pd, ISPC, 2);

class mm_storeu_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                _mm_storeu_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storeu_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_storeu_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_pd, ISPC, 2);

class mm_storer_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1];
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                _mm_storer_pd(&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storer_pd(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] && Result[k + 1] != Source1[k]) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_storer_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storer_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storer_pd, ISPC, 2);

class mm_storeh_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0;
            Source1[k] = (double)k;
            Source2[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1] + Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                _mm_storeh_pd(&Result[k], S);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storeh_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] + Source2[k + 1]) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_storeh_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeh_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeh_pd, ISPC, 2);

class mm_storel_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0.0f;
            Source1[k] = (double)k;
            Source2[k] = (double)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k] + Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d S2 = _mm_loadu_pd(&Source2[k]);
                const __m128d S = _mm_add_pd(S1, S2);
                _mm_storel_pd(&Result[k], S);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storel_pd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
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

    double *Result;
    double *Source1;
    double *Source2;
};

BENCHMARK_CASE_POW2(mm_storel_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storel_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storel_pd, ISPC, 2);

class mm_add_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)(k % 63);
            Source2[k] = (int8)((Iterations - k) % 63);
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
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_add_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_epi8(Result, Source1, Source2, Iterations);
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_add_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi8, ISPC, 2);

class mm_add_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
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
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_add_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_epi16(Result, Source1, Source2, Iterations);
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_add_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi16, ISPC, 2);

class mm_add_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new int[Iterations];
        Source2 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = k;
            Source2[k] = (Iterations - k);
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_add_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_epi32(Result, Source1, Source2, Iterations);
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

    int *Result;
    int *Source1;
    int *Source2;
};

BENCHMARK_CASE_POW2(mm_add_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi32, ISPC, 2);

class mm_add_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];
        Source2 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = k;
            Source2[k] = (Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_add_epi64(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_add_epi64(Result, Source1, Source2, Iterations);
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

    int64 *Result;
    int64 *Source1;
    int64 *Source2;
};

BENCHMARK_CASE_POW2(mm_add_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_add_epi64, ISPC, 2);

class mm_adds_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)(k % 256);
            Source2[k] = (int8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint8 a_unsig = Source1[k], b_unsig = Source2[k];
                uint8 result = a_unsig + b_unsig;
                a_unsig = (a_unsig >> 7) + INT8_MAX;
                if ((int8)((a_unsig ^ b_unsig) | ~(b_unsig ^ result)) >= 0)
                    result = a_unsig;
                Result[k] = (int8)result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_adds_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_adds_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint8 a_unsig = Source1[k], b_unsig = Source2[k];
            uint8 result = a_unsig + b_unsig;
            a_unsig = (a_unsig >> 7) + INT8_MAX;
            if ((int8)((a_unsig ^ b_unsig) | ~(b_unsig ^ result)) >= 0)
                result = a_unsig;

            if (Result[k] != (int8)result) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_adds_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epi8, ISPC, 2);

class mm_adds_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k);
            Source2[k] = (int16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint16 a_unsig = Source1[k], b_unsig = Source2[k];
                uint16 result = a_unsig + b_unsig;
                a_unsig = (a_unsig >> 15) + INT16_MAX;
                if ((int16)((a_unsig ^ b_unsig) | ~(b_unsig ^ result)) >= 0)
                    result = a_unsig;
                Result[k] = (int16)result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_adds_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_adds_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint16 a_unsig = Source1[k], b_unsig = Source2[k];
            uint16 result = a_unsig + b_unsig;
            a_unsig = (a_unsig >> 15) + INT16_MAX;
            if ((int16)((a_unsig ^ b_unsig) | ~(b_unsig ^ result)) >= 0)
                result = a_unsig;

            if (Result[k] != (int16)result) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_adds_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epi16, ISPC, 2);

class mm_adds_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint8 result = Source1[k] + Source2[k];
                result |= (-(uint8)(result < Source1[k]));
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_adds_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_adds_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint8 result = Source1[k] + Source2[k];
            result |= (-(uint8)(result < Source1[k]));

            if (Result[k] != result) {
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

    uint8 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_adds_epu8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epu8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epu8, ISPC, 2);

class mm_adds_epu16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];
        Source2 = new uint16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint16)(k);
            Source2[k] = (uint16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint16 result = Source1[k] + Source2[k];
                result |= (-(uint16)(result < Source1[k]));
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_adds_epu16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_adds_epu16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint16 result = Source1[k] + Source2[k];
            result |= (-(uint16)(result < Source1[k]));

            if (Result[k] != result) {
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

    uint16 *Result;
    uint16 *Source1;
    uint16 *Source2;
};

BENCHMARK_CASE_POW2(mm_adds_epu16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epu16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_adds_epu16, ISPC, 2);

class mm_avg_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint8 result = (Source1[k] + Source2[k] + 1) >> 1;
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_avg_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_avg_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint8 result = (Source1[k] + Source2[k] + 1) >> 1;

            if (Result[k] != result) {
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

    uint8 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_avg_epu8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_avg_epu8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_avg_epu8, ISPC, 2);

class mm_avg_epu16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];
        Source2 = new uint16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint16)(k);
            Source2[k] = (uint16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint16 result = (Source1[k] + Source2[k] + 1) >> 1;
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_avg_epu16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_avg_epu16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint16 result = (Source1[k] + Source2[k] + 1) >> 1;

            if (Result[k] != result) {
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

    uint16 *Result;
    uint16 *Source1;
    uint16 *Source2;
};

BENCHMARK_CASE_POW2(mm_avg_epu16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_avg_epu16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_avg_epu16, ISPC, 2);

class mm_madd_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k);
            Source2[k] = (int16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k] * Source2[k] + Source1[k + 1] * Source2[k + 1];
                Result[k + 1] = Source1[k + 2] * Source2[k + 2] + Source1[k + 3] * Source2[k + 3];
                Result[k + 2] = Source1[k + 4] * Source2[k + 4] + Source1[k + 5] * Source2[k + 5];
                Result[k + 3] = Source1[k + 6] * Source2[k + 6] + Source1[k + 7] * Source2[k + 7];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_madd_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_madd_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            int32 Result0 = Source1[k] * Source2[k] + Source1[k + 1] * Source2[k + 1];
            int32 Result1 = Source1[k + 2] * Source2[k + 2] + Source1[k + 3] * Source2[k + 3];
            int32 Result2 = Source1[k + 4] * Source2[k + 4] + Source1[k + 5] * Source2[k + 5];
            int32 Result3 = Source1[k + 6] * Source2[k + 6] + Source1[k + 7] * Source2[k + 7];

            if (Result[k] != Result0 && Result[k + 1] != Result1 && Result[k + 2] != Result2 &&
                Result[k + 3] != Result3) {
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

    int32 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_madd_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_madd_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_madd_epi16, ISPC, 2);

class mm_max_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? Source1[k] : Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_max_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] > Source2[k] ? Source1[k] : Source2[k])) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_max_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_epi16, ISPC, 2);

class mm_max_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? Source1[k] : Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_max_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_max_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] > Source2[k] ? Source1[k] : Source2[k])) {
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

    uint8 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_max_epu8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_epu8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_max_epu8, ISPC, 2);

class mm_min_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? Source1[k] : Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_min_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] < Source2[k] ? Source1[k] : Source2[k])) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_min_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_epi16, ISPC, 2);

class mm_min_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? Source1[k] : Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_min_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_min_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] < Source2[k] ? Source1[k] : Source2[k])) {
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

    uint8 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_min_epu8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_epu8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_min_epu8, ISPC, 2);

class mm_mulhi_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int S = Source1[k] * Source2[k];
                S >>= 16;
                Result[k] = *(reinterpret_cast<int16 *>(&S));
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_mulhi_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mulhi_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            int S = Source1[k] * Source2[k];
            S >>= 16;
            const int16 result = *(reinterpret_cast<int16 *>(&S));

            if (Result[k] != result) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_mulhi_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mulhi_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mulhi_epi16, ISPC, 2);

class mm_mulhi_epu16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];
        Source2 = new uint16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint32 S = Source1[k] * Source2[k];
                S >>= 16;
                Result[k] = *(reinterpret_cast<uint16 *>(&S));
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_mulhi_epu16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mulhi_epu16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            uint32 S = Source1[k] * Source2[k];
            S >>= 16;
            const uint16 result = *(reinterpret_cast<uint16 *>(&S));

            if (Result[k] != result) {
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

    uint16 *Result;
    uint16 *Source1;
    uint16 *Source2;
};

BENCHMARK_CASE_POW2(mm_mulhi_epu16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mulhi_epu16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mulhi_epu16, ISPC, 2);

class mm_mullo_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                int S = Source1[k] * Source2[k];
                Result[k] = static_cast<int16>(S);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_mullo_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mullo_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            int S = Source1[k] * Source2[k];
            const int16 result = static_cast<int16>(S);

            if (Result[k] != result) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_mullo_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mullo_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mullo_epi16, ISPC, 2);

class mm_mul_epu32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint64[Iterations];
        Source1 = new uint32[Iterations];
        Source2 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = k;
            Source2[k] = (Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = static_cast<uint64>(Source1[k]) * static_cast<uint64>(Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_mul_epu32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_mul_epu32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != static_cast<uint64>(Source1[k]) * static_cast<uint64>(Source2[k])) {
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

    uint64 *Result;
    uint32 *Source1;
    uint32 *Source2;
};

BENCHMARK_CASE_POW2(mm_mul_epu32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_epu32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_mul_epu32, ISPC, 2);

#ifdef IS_X86_ARCH
class mm_sad_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sad_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sad_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sad_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_sad_epu8(S1, S2);

            if (memcmp(&Result[k], &R, sizeof(__m128i)) != 0) {
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

    uint16 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_sad_epu8, CPP, 0);
BENCHMARK_CASE_POW2(mm_sad_epu8, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sad_epu8, ISPC, 2);
#endif // IS_X86_ARCH

class mm_sub_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)(k % 63);
            Source2[k] = (int8)((Iterations - k) % 63);
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
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sub_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_epi8(Result, Source1, Source2, Iterations);
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi8, ISPC, 2);

class mm_sub_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k % 16383);
            Source2[k] = (int16)((Iterations - k) % 16383);
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
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sub_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_epi16(Result, Source1, Source2, Iterations);
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi16, ISPC, 2);

class mm_sub_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int[Iterations];
        Source1 = new int[Iterations];
        Source2 = new int[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = k;
            Source2[k] = (Iterations - k);
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sub_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_epi32(Result, Source1, Source2, Iterations);
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

    int *Result;
    int *Source1;
    int *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi32, ISPC, 2);

class mm_sub_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];
        Source2 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = k;
            Source2[k] = (Iterations - k);
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_sub_epi64(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_sub_epi64(Result, Source1, Source2, Iterations);
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

    int64 *Result;
    int64 *Source1;
    int64 *Source2;
};

BENCHMARK_CASE_POW2(mm_sub_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_sub_epi64, ISPC, 2);

class mm_subs_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)(k % 256);
            Source2[k] = (int8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint8 a_unsig = Source1[k], b_unsig = Source2[k];
                uint8 result = a_unsig - b_unsig;
                a_unsig = (a_unsig >> 7) + INT8_MAX;
                if ((int8)((a_unsig ^ b_unsig) & (a_unsig ^ result)) < 0)
                    result = a_unsig;
                Result[k] = (int8)result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_subs_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_subs_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint8 a_unsig = Source1[k], b_unsig = Source2[k];
            uint8 result = a_unsig - b_unsig;
            a_unsig = (a_unsig >> 7) + INT8_MAX;
            if ((int8)((a_unsig ^ b_unsig) & (a_unsig ^ result)) < 0)
                result = a_unsig;

            if (Result[k] != (int8)result) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_subs_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epi8, ISPC, 2);

class mm_subs_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k);
            Source2[k] = (int16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint16 a_unsig = Source1[k], b_unsig = Source2[k];
                uint16 result = a_unsig - b_unsig;
                a_unsig = (a_unsig >> 15) + INT16_MAX;
                if ((int16)((a_unsig ^ b_unsig) & (a_unsig ^ result)) < 0)
                    result = a_unsig;
                Result[k] = (int16)result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_subs_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_subs_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint16 a_unsig = Source1[k], b_unsig = Source2[k];
            uint16 result = a_unsig - b_unsig;
            a_unsig = (a_unsig >> 15) + INT16_MAX;
            if ((int16)((a_unsig ^ b_unsig) & (a_unsig ^ result)) < 0)
                result = a_unsig;

            if (Result[k] != (int16)result) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_subs_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epi16, ISPC, 2);

class mm_subs_epu8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new uint8[Iterations];
        Source2 = new uint8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint8)(k % 256);
            Source2[k] = (uint8)((Iterations - k) % 256);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint8 result = Source1[k] - Source2[k];
                result &= (-(uint8)(result <= Source1[k]));
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_subs_epu8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_subs_epu8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint8 result = Source1[k] - Source2[k];
            result &= (-(uint8)(result <= Source1[k]));

            if (Result[k] != result) {
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

    uint8 *Result;
    uint8 *Source1;
    uint8 *Source2;
};

BENCHMARK_CASE_POW2(mm_subs_epu8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epu8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epu8, ISPC, 2);

class mm_subs_epu16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];
        Source2 = new uint16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint16)(k);
            Source2[k] = (uint16)((Iterations - k));
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                uint16 result = Source1[k] - Source2[k];
                result &= (-(uint16)(result <= Source1[k]));
                Result[k] = result;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_subs_epu16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_subs_epu16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            uint16 result = Source1[k] - Source2[k];
            result &= (-(uint16)(result <= Source1[k]));

            if (Result[k] != result) {
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

    uint16 *Result;
    uint16 *Source1;
    uint16 *Source2;
};

BENCHMARK_CASE_POW2(mm_subs_epu16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epu16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_subs_epu16, ISPC, 2);

class mm_and_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];
        Source2 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
            Source2[k] = (uint32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (Source1[k] & Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_and_si128(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_and_si128(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] & Source2[k])) {
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

    uint32 *Result;
    uint32 *Source1;
    uint32 *Source2;
};

BENCHMARK_CASE_POW2(mm_and_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_and_si128, ISPC, 2);

class mm_andnot_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];
        Source2 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
            Source2[k] = (uint32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = ((~(Source1[k])) & Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_andnot_si128(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_andnot_si128(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != ((~(Source1[k])) & Source2[k])) {
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

    uint32 *Result;
    uint32 *Source1;
    uint32 *Source2;
};

BENCHMARK_CASE_POW2(mm_andnot_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_andnot_si128, ISPC, 2);

class mm_or_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];
        Source2 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
            Source2[k] = (uint32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (Source1[k] | Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_or_si128(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_or_si128(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] | Source2[k])) {
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

    uint32 *Result;
    uint32 *Source1;
    uint32 *Source2;
};

BENCHMARK_CASE_POW2(mm_or_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_or_si128, ISPC, 2);

class mm_xor_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];
        Source2 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
            Source2[k] = (uint32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = (Source1[k] ^ Source2[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_xor_si128(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_xor_si128(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != (Source1[k] ^ Source2[k])) {
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

    uint32 *Result;
    uint32 *Source1;
    uint32 *Source2;
};

BENCHMARK_CASE_POW2(mm_xor_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_xor_si128, ISPC, 2);

static const int LeftShift = 8;
static const int RightShift = 4;

#ifdef IS_X86_ARCH
class mm_slli_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                uint8 *S = (uint8 *)&Source1[k];
                uint8 *R = (uint8 *)&Result[k];

                const int _Imm = LeftShift & 0xF;

                for (int i = 0; i < 16; i++) {
                    R[i] = i < (_Imm) ? 0 : S[i - _Imm];
                }
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_slli_si128(S1, LeftShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_slli_si128(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_slli_si128(S1, LeftShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint32 *Result;
    uint32 *Source1;
};

BENCHMARK_CASE_POW2(mm_slli_si128, CPP, 0);
BENCHMARK_CASE_POW2(mm_slli_si128, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_slli_si128, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_slli_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = LeftShift & 0xFF;
                Result[k] = (_Count > 15) ? 0 : Source1[k] << _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_slli_epi16(S1, LeftShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_slli_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_slli_epi16(S1, LeftShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_slli_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_slli_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_slli_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_sll_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, LeftShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 15) ? 0 : Source1[k] << _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_sll_epi16(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sll_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_sll_epi16(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_sll_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_sll_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sll_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_slli_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = LeftShift & 0xFF;
                Result[k] = (_Count > 31) ? 0 : Source1[k] << _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_slli_epi32(S1, LeftShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_slli_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_slli_epi32(S1, LeftShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_slli_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_slli_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_slli_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_sll_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, LeftShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 31) ? 0 : Source1[k] << _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_sll_epi32(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sll_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_sll_epi32(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_sll_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_sll_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sll_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_slli_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = LeftShift & 0xFF;
                Result[k] = (_Count > 63) ? 0 : Source1[k] << _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_slli_epi64(S1, LeftShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_slli_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_slli_epi64(S1, LeftShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_slli_epi64, CPP, 0);
BENCHMARK_CASE_POW2(mm_slli_epi64, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_slli_epi64, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_sll_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, LeftShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 63) ? 0 : Source1[k] << _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_sll_epi64(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sll_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_sll_epi64(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int64 *Result;
    int64 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_sll_epi64, CPP, 0);
BENCHMARK_CASE_POW2(mm_sll_epi64, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sll_epi64, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srai_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = RightShift & 0xFF;
                Result[k] = (_Count > 15) ? (Source1[k] ? 0xFFFF : 0) : Source1[k] >> _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srai_epi16(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srai_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srai_epi16(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_srai_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_srai_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srai_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_sra_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, RightShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 15) ? (Source1[k] ? 0xFFFF : 0) : Source1[k] >> _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_sra_epi16(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sra_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_sra_epi16(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_sra_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_sra_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sra_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srai_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = RightShift & 0xFF;
                Result[k] = (_Count > 31) ? (Source1[k] ? 0xFFFFFFFF : 0) : Source1[k] >> _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srai_epi32(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srai_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srai_epi32(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_srai_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_srai_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srai_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_sra_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, RightShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 31) ? (Source1[k] ? 0xFFFFFFFF : 0) : Source1[k] >> _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_sra_epi32(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_sra_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_sra_epi32(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_sra_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_sra_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_sra_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srli_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                uint8 *S = (uint8 *)&Source1[k];
                uint8 *R = (uint8 *)&Result[k];

                const int _Imm = RightShift & 0xF;

                for (int i = 0; i < 16; i++) {
                    R[i] = 16 - i > (_Imm) ? S[i + _Imm] : 0;
                }
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srli_si128(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srli_si128(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srli_si128(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint32 *Result;
    uint32 *Source1;
};

BENCHMARK_CASE_POW2(mm_srli_si128, CPP, 0);
BENCHMARK_CASE_POW2(mm_srli_si128, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srli_si128, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srli_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = RightShift & 0xFF;
                Result[k] = (_Count > 15) ? (Source1[k] ? 0xFFFF : 0) : Source1[k] >> _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srli_epi16(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srli_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srli_epi16(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint16 *Result;
    uint16 *Source1;
};

BENCHMARK_CASE_POW2(mm_srli_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_srli_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srli_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srl_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint16[Iterations];
        Source1 = new uint16[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, RightShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 15) ? (Source1[k] ? 0xFFFF : 0) : Source1[k] >> _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srl_epi16(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srl_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srl_epi16(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint16 *Result;
    uint16 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_srl_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_srl_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srl_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srli_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = RightShift & 0xFF;
                Result[k] = (_Count > 31) ? (Source1[k] ? 0xFFFFFFFF : 0) : Source1[k] >> _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srli_epi32(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srli_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srli_epi32(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint32 *Result;
    uint32 *Source1;
};

BENCHMARK_CASE_POW2(mm_srli_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_srli_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srli_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srl_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint32[Iterations];
        Source1 = new uint32[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, RightShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 31) ? (Source1[k] ? 0xFFFFFFFF : 0) : Source1[k] >> _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srl_epi32(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srl_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srl_epi32(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint32 *Result;
    uint32 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_srl_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_srl_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srl_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srli_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint64[Iterations];
        Source1 = new uint64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const int _Count = RightShift & 0xFF;
                Result[k] = (_Count > 63) ? (Source1[k] ? 0xFFFFFFFFFFFFFFFF : 0) : Source1[k] >> _Count;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srli_epi64(S1, RightShift);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srli_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srli_epi64(S1, RightShift);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint64 *Result;
    uint64 *Source1;
};

BENCHMARK_CASE_POW2(mm_srli_epi64, CPP, 0);
BENCHMARK_CASE_POW2(mm_srli_epi64, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srli_epi64, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_srl_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint64[Iterations];
        Source1 = new uint64[Iterations];
        _Count = _mm_set_epi32(0, 0, 0, RightShift);

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (uint64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                const uint64 _Imm = *((uint64 *)&_Count);
                Result[k] = (_Imm > 63) ? (Source1[k] ? 0xFFFFFFFFFFFFFFFF : 0) : Source1[k] >> _Imm;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_srl_epi64(S1, _Count);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_srl_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i R = _mm_srl_epi64(S1, _Count);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    uint64 *Result;
    uint64 *Source1;
    int32 pad[2];
    __m128i _Count;
};

BENCHMARK_CASE_POW2(mm_srl_epi64, CPP, 0);
BENCHMARK_CASE_POW2(mm_srl_epi64, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_srl_epi64, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpeq_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Source2[k] = (int8)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? 0xFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpeq_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpeq_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpeq_epi8(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_epi8, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpeq_epi8, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpeq_epi8, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpeq_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? 0xFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpeq_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpeq_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpeq_epi16(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpeq_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpeq_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpeq_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] == Source2[k] ? 0xFFFFFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpeq_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpeq_epi32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpeq_epi32(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpeq_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpeq_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpeq_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpgt_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Source2[k] = (int8)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? 0xFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpgt_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpgt_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpgt_epi8(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_epi8, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpgt_epi8, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpgt_epi8, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpgt_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? 0xFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpgt_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpgt_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpgt_epi16(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpgt_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpgt_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmpgt_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] > Source2[k] ? 0xFFFFFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmpgt_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmpgt_epi32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmpgt_epi32(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmpgt_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmpgt_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmpgt_epi32, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmplt_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Source2[k] = (int8)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? 0xFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmplt_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmplt_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmplt_epi8(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_epi8, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmplt_epi8, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmplt_epi8, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmplt_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? 0xFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmplt_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmplt_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmplt_epi16(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_epi16, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmplt_epi16, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmplt_epi16, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_cmplt_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k] < Source2[k] ? 0xFFFFFFFF : 0;
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_cmplt_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
        case 2:
            ispc::mm_cmplt_epi32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            const __m128i R0 = _mm_loadu_si128((__m128i *)&Result[k]);
            const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
            const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
            const __m128i R = _mm_cmplt_epi32(S1, S2);

            if (memcmp(&R0, &R, sizeof(__m128i)) != 0) {
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_cmplt_epi32, CPP, 0);
BENCHMARK_CASE_POW2(mm_cmplt_epi32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_cmplt_epi32, ISPC, 2);
#endif // IS_X86_ARCH

class mm_cvtsi32_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k - (Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = 0;
                Result[k + 2] = 0;
                Result[k + 3] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_cvtsi32_si128(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi32_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsi32_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi32_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi32_si128, ISPC, 2);

class mm_cvtsi128_si32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                Result[k] = _mm_cvtsi128_si32(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi128_si32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0 && Result[k + 2] != 0 && Result[k + 3] != 0) {
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsi128_si32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi128_si32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi128_si32, ISPC, 2);

static inline int8 Saturate8(int16 _A) { return _A < INT8_MIN ? INT8_MIN : _A > INT8_MAX ? INT8_MAX : (int8)_A; }

class mm_packs_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Saturate8(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source1[k + 8]);
                const __m128i R = _mm_packs_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_packs_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Saturate8(Source1[k])) {
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

    int8 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_packs_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packs_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packs_epi16, ISPC, 2);

static inline int16 Saturate16(int32 _A) { return _A < INT16_MIN ? INT16_MIN : _A > INT16_MAX ? INT16_MAX : (int16)_A; }

class mm_packs_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Saturate16(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source1[k + 4]);
                const __m128i R = _mm_packs_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_packs_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != Saturate16(Source1[k])) {
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

    int16 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_packs_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packs_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packs_epi32, ISPC, 2);

static inline uint8 SaturateU8(int16 _A) { return _A < 0 ? 0 : _A > UINT8_MAX ? UINT8_MAX : (uint8)_A; }

class mm_packus_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new uint8[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = SaturateU8(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source1[k + 8]);
                const __m128i R = _mm_packus_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_packus_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != SaturateU8(Source1[k])) {
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

    uint8 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_packus_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packus_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_packus_epi16, ISPC, 2);

class mm_extract_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 1;
            Source1[k] = (int16)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k + 4];
                Result[k + 1] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                *((int *)&Result[k]) = _mm_extract_epi16(S1, 4);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_extract_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            if (Result[k] != Source1[k + 4] && Result[k + 1] != 0) {
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_extract_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_extract_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_extract_epi16, ISPC, 2);

class mm_insert_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                for (int i = 0; i < 8; i++) {
                    Result[k + i] = i == 4 ? 1 : Source1[k + i];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                __m128i R = _mm_insert_epi16(S1, 1, 4);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_insert_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            for (int i = 0; i < 8; i++) {
                if (i == 4) {
                    if (Result[k + i] != 1) {
                        return false;
                    }
                } else {
                    if (Result[k + i] != Source1[k + i]) {
                        return false;
                    }
                }
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_insert_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_insert_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_insert_epi16, ISPC, 2);

class mm_movemask_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int8)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 16) {
                int R = 0;
                for (int i = 0; i < 16; i++) {
                    R |= (Source1[k] & 0x80 ? 1 << i : 0);
                }
                Result[k] = R;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                int R = _mm_movemask_epi8(S1);
                Result[k] = R;
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_movemask_epi8(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            int R = 0;
            for (int i = 0; i < 16; i++) {
                R |= (Source1[k] & 0x80 ? 1 << i : 0);
            }
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

    int32 *Result;
    int8 *Source1;
};

BENCHMARK_CASE_POW2(mm_movemask_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_movemask_epi8, ISPC, 2);

class mm_shuffle_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i S = _mm_add_epi32(S1, S2);
                const __m128i R = _mm_shuffle_epi32(S, 0x1B);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_shuffle_epi32(Result, Source1, Source2, Iterations);
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_shuffle_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shuffle_epi32, ISPC, 2);

class mm_shufflehi_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k] + Source2[k];
                Result[k + 1] = Source1[k + 1] + Source2[k + 1];
                Result[k + 2] = Source1[k + 2] + Source2[k + 2];
                Result[k + 3] = Source1[k + 3] + Source2[k + 3];
                Result[k + 4] = Source1[k + 7] + Source2[k + 7];
                Result[k + 5] = Source1[k + 6] + Source2[k + 6];
                Result[k + 6] = Source1[k + 5] + Source2[k + 5];
                Result[k + 7] = Source1[k + 4] + Source2[k + 4];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i S = _mm_add_epi16(S1, S2);
                const __m128i R = _mm_shufflehi_epi16(S, 0x1B);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_shufflehi_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            if (Result[k] != Source1[k] + Source2[k] && Result[k + 1] != Source1[k + 1] + Source2[k + 1] &&
                Result[k + 2] != Source1[k + 2] + Source2[k + 2] && Result[k + 3] != Source1[k + 3] + Source2[k + 3] &&
                Result[k + 4] != Source1[k + 7] + Source2[k + 7] && Result[k + 5] != Source1[k + 6] + Source2[k + 6] &&
                Result[k + 6] != Source1[k + 5] + Source2[k + 5] && Result[k + 7] != Source1[k + 4] + Source2[k + 4]) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_shufflehi_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shufflehi_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shufflehi_epi16, ISPC, 2);

class mm_shufflelo_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k + 3] + Source2[k + 3];
                Result[k + 1] = Source1[k + 2] + Source2[k + 2];
                Result[k + 2] = Source1[k + 1] + Source2[k + 1];
                Result[k + 3] = Source1[k] + Source2[k];
                Result[k + 4] = Source1[k + 4] + Source2[k + 4];
                Result[k + 5] = Source1[k + 5] + Source2[k + 5];
                Result[k + 6] = Source1[k + 6] + Source2[k + 6];
                Result[k + 7] = Source1[k + 7] + Source2[k + 7];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i S = _mm_add_epi16(S1, S2);
                const __m128i R = _mm_shufflelo_epi16(S, 0x1B);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_shufflehi_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            if (Result[k] != Source1[k + 3] + Source2[k + 3] && Result[k + 1] != Source1[k + 2] + Source2[k + 2] &&
                Result[k + 2] != Source1[k + 1] + Source2[k + 1] && Result[k + 3] != Source1[k] + Source2[k] &&
                Result[k + 4] != Source1[k + 4] + Source2[k + 4] && Result[k + 5] != Source1[k + 5] + Source2[k + 5] &&
                Result[k + 6] != Source1[k + 6] + Source2[k + 6] && Result[k + 7] != Source1[k + 7] + Source2[k + 7]) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_shufflelo_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shufflelo_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_shufflelo_epi16, ISPC, 2);

class mm_unpackhi_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Source2[k] = (int8)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 16) {
                Result[k] = Source1[k + 8];
                Result[k + 1] = Source2[k + 8];
                Result[k + 2] = Source1[k + 9];
                Result[k + 3] = Source2[k + 9];
                Result[k + 4] = Source1[k + 10];
                Result[k + 5] = Source2[k + 10];
                Result[k + 6] = Source1[k + 11];
                Result[k + 7] = Source2[k + 11];
                Result[k + 8] = Source1[k + 12];
                Result[k + 9] = Source2[k + 12];
                Result[k + 10] = Source1[k + 13];
                Result[k + 11] = Source2[k + 13];
                Result[k + 12] = Source1[k + 14];
                Result[k + 13] = Source2[k + 14];
                Result[k + 14] = Source1[k + 15];
                Result[k + 15] = Source2[k + 15];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpackhi_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            if (Result[k] != Source1[k + 8] && Result[k + 1] != Source2[k + 8] && Result[k + 2] != Source1[k + 9] &&
                Result[k + 3] != Source2[k + 9] && Result[k + 4] != Source1[k + 10] &&
                Result[k + 5] != Source2[k + 10] && Result[k + 6] != Source1[k + 11] &&
                Result[k + 7] != Source2[k + 11] && Result[k + 8] != Source1[k + 12] &&
                Result[k + 9] != Source2[k + 12] && Result[k + 10] != Source1[k + 13] &&
                Result[k + 11] != Source2[k + 13] && Result[k + 12] != Source1[k + 14] &&
                Result[k + 13] != Source2[k + 14] && Result[k + 14] != Source1[k + 15] &&
                Result[k + 15] != Source2[k + 15]) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi8, ISPC, 2);

class mm_unpackhi_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k + 4];
                Result[k + 1] = Source2[k + 4];
                Result[k + 2] = Source1[k + 5];
                Result[k + 3] = Source2[k + 5];
                Result[k + 4] = Source1[k + 6];
                Result[k + 5] = Source2[k + 6];
                Result[k + 6] = Source1[k + 7];
                Result[k + 7] = Source2[k + 7];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpackhi_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            if (Result[k] != Source1[k + 4] && Result[k + 1] != Source2[k + 4] && Result[k + 2] != Source1[k + 5] &&
                Result[k + 3] != Source2[k + 5] && Result[k + 4] != Source1[k + 6] && Result[k + 5] != Source2[k + 6] &&
                Result[k + 6] != Source1[k + 7] && Result[k + 7] != Source2[k + 7]) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi16, ISPC, 2);

class mm_unpackhi_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k + 2];
                Result[k + 1] = Source2[k + 2];
                Result[k + 2] = Source1[k + 3];
                Result[k + 3] = Source2[k + 3];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpackhi_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_epi32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k + 2] && Result[k + 1] != Source2[k + 2] && Result[k + 2] != Source1[k + 3] &&
                Result[k + 3] != Source2[k + 3]) {
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi32, ISPC, 2);

class mm_unpackhi_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];
        Source2 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int64)k;
            Source2[k] = (int64)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k + 1];
                Result[k + 1] = Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpackhi_epi64(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpackhi_epi64(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k + 1] && Result[k + 1] != Source2[k + 1]) {
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

    int64 *Result;
    int64 *Source1;
    int64 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpackhi_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpackhi_epi64, ISPC, 2);

class mm_unpacklo_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Source2 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Source2[k] = (int8)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 16) {
                Result[k] = Source1[k];
                Result[k + 1] = Source2[k];
                Result[k + 2] = Source1[k + 1];
                Result[k + 3] = Source2[k + 1];
                Result[k + 4] = Source1[k + 2];
                Result[k + 5] = Source2[k + 2];
                Result[k + 6] = Source1[k + 3];
                Result[k + 7] = Source2[k + 3];
                Result[k + 8] = Source1[k + 4];
                Result[k + 9] = Source2[k + 4];
                Result[k + 10] = Source1[k + 5];
                Result[k + 11] = Source2[k + 5];
                Result[k + 12] = Source1[k + 6];
                Result[k + 13] = Source2[k + 6];
                Result[k + 14] = Source1[k + 7];
                Result[k + 15] = Source2[k + 7];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpacklo_epi8(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_epi8(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source2[k] && Result[k + 2] != Source1[k + 1] &&
                Result[k + 3] != Source2[k + 1] && Result[k + 4] != Source1[k + 2] && Result[k + 5] != Source2[k + 2] &&
                Result[k + 6] != Source1[k + 3] && Result[k + 7] != Source2[k + 3] && Result[k + 8] != Source1[k + 4] &&
                Result[k + 9] != Source2[k + 4] && Result[k + 10] != Source1[k + 5] &&
                Result[k + 11] != Source2[k + 5] && Result[k + 12] != Source1[k + 6] &&
                Result[k + 13] != Source2[k + 6] && Result[k + 14] != Source1[k + 7] &&
                Result[k + 15] != Source2[k + 7]) {
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

    int8 *Result;
    int8 *Source1;
    int8 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi8, ISPC, 2);

class mm_unpacklo_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];
        Source2 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int16)k;
            Source2[k] = (int16)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                Result[k] = Source1[k];
                Result[k + 1] = Source2[k];
                Result[k + 2] = Source1[k + 1];
                Result[k + 3] = Source2[k + 1];
                Result[k + 4] = Source1[k + 2];
                Result[k + 5] = Source2[k + 2];
                Result[k + 6] = Source1[k + 3];
                Result[k + 7] = Source2[k + 4];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpacklo_epi16(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_epi16(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source2[k] && Result[k + 2] != Source1[k + 1] &&
                Result[k + 3] != Source2[k + 1] && Result[k + 4] != Source1[k + 2] && Result[k + 5] != Source2[k + 2] &&
                Result[k + 6] != Source1[k + 3] && Result[k + 7] != Source2[k + 3]) {
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

    int16 *Result;
    int16 *Source1;
    int16 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi16, ISPC, 2);

class mm_unpacklo_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];
        Source2 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
            Source2[k] = (int32)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                Result[k] = Source1[k];
                Result[k + 1] = Source2[k];
                Result[k + 2] = Source1[k + 1];
                Result[k + 3] = Source2[k + 1];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpacklo_epi32(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_epi32(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source2[k] && Result[k + 2] != Source1[k + 1] &&
                Result[k + 3] != Source2[k + 1]) {
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

    int32 *Result;
    int32 *Source1;
    int32 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi32, ISPC, 2);

class mm_unpacklo_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];
        Source2 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int64)k;
            Source2[k] = (int64)(Iterations - k);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = Source2[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i S2 = _mm_loadu_si128((__m128i *)&Source2[k]);
                const __m128i R = _mm_unpacklo_epi64(S1, S2);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_unpacklo_epi64(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != Source2[k]) {
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

    int64 *Result;
    int64 *Source1;
    int64 *Source2;
};

BENCHMARK_CASE_POW2(mm_unpacklo_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_unpacklo_epi64, ISPC, 2);

class mm_load_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128i S1 = _mm_load_si128((__m128i *)&Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_load_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_load_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_load_si128, ISPC, 2);

class mm_loadu_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadu_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_loadu_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadu_si128, ISPC, 2);

class mm_loadl_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadl_epi64((__m128i *)&Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_loadl_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_loadl_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadl_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_loadl_epi64, ISPC, 2);

class mm_set_epi64x : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)k;
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i R = _mm_set_epi64x(Source1[k + 1], Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_epi64x(Result, Source1, Iterations);
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_set_epi64x, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi64x, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi64x, ISPC, 2);

class mm_set_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int32)k;
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
                const __m128i R = _mm_set_epi32(Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_epi32(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_set_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi32, ISPC, 2);

class mm_set_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int16)k;
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
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i R = _mm_set_epi16(Source1[k + 7], Source1[k + 6], Source1[k + 5], Source1[k + 4],
                                                Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_epi16(Result, Source1, Iterations);
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_set_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi16, ISPC, 2);

class mm_set_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int8)k;
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
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i R = _mm_set_epi8(Source1[k + 15], Source1[k + 14], Source1[k + 13], Source1[k + 12],
                                               Source1[k + 11], Source1[k + 10], Source1[k + 9], Source1[k + 8],
                                               Source1[k + 7], Source1[k + 6], Source1[k + 5], Source1[k + 4],
                                               Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set_epi8(Result, Source1, Iterations);
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

    int8 *Result;
    int8 *Source1;
};

BENCHMARK_CASE_POW2(mm_set_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set_epi8, ISPC, 2);

class mm_set1_epi64x : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                for (int i = 0; i < 2; i++) {
                    Result[k + i] = Source1[k];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i R = _mm_set1_epi64x(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set1_epi64x(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            for (int i = 0; i < 2; i++) {
                if (Result[k + i] != Source1[k]) {
                    return false;
                }
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_set1_epi64x, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi64x, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi64x, ISPC, 2);

class mm_set1_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 4) {
                for (int i = 0; i < 4; i++) {
                    Result[k + i] = Source1[k];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i R = _mm_set1_epi32(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set1_epi32(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 4) {
            for (int i = 0; i < 4; i++) {
                if (Result[k + i] != Source1[k]) {
                    return false;
                }
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_set1_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi32, ISPC, 2);

class mm_set1_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int16)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 8) {
                for (int i = 0; i < 8; i++) {
                    Result[k + i] = Source1[k];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i R = _mm_set1_epi16(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set1_epi16(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 8) {
            for (int i = 0; i < 8; i++) {
                if (Result[k + i] != Source1[k]) {
                    return false;
                }
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_set1_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi16, ISPC, 2);

class mm_set1_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int8)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 16) {
                for (int i = 0; i < 16; i++) {
                    Result[k + i] = Source1[k];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i R = _mm_set1_epi8(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_set1_epi8(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 16) {
            for (int i = 0; i < 16; i++) {
                if (Result[k + i] != Source1[k]) {
                    return false;
                }
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

    int8 *Result;
    int8 *Source1;
};

BENCHMARK_CASE_POW2(mm_set1_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_set1_epi8, ISPC, 2);

class mm_setr_epi32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int32)k;
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
                const __m128i R = _mm_setr_epi32(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setr_epi32(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_setr_epi32, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi32, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi32, ISPC, 2);

class mm_setr_epi16 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int16[Iterations];
        Source1 = new int16[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int16)k;
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
            for (int k = 0; k < Iterations; k += 8) {
                const __m128i R = _mm_setr_epi16(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3],
                                                 Source1[k + 4], Source1[k + 5], Source1[k + 6], Source1[k + 7]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setr_epi16(Result, Source1, Iterations);
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

    int16 *Result;
    int16 *Source1;
};

BENCHMARK_CASE_POW2(mm_setr_epi16, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi16, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi16, ISPC, 2);

class mm_setr_epi8 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int8)k;
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
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i R = _mm_setr_epi8(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3],
                                                Source1[k + 4], Source1[k + 5], Source1[k + 6], Source1[k + 7],
                                                Source1[k + 8], Source1[k + 9], Source1[k + 10], Source1[k + 11],
                                                Source1[k + 12], Source1[k + 13], Source1[k + 14], Source1[k + 15]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setr_epi8(Result, Source1, Iterations);
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

    int8 *Result;
    int8 *Source1;
};

BENCHMARK_CASE_POW2(mm_setr_epi8, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi8, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setr_epi8, ISPC, 2);

class mm_setzero_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 1;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i R = _mm_setzero_si128();
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_setzero_si128(Result, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if (Result[k] != 0) {
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

    int32 *Result;
};

BENCHMARK_CASE_POW2(mm_setzero_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_setzero_si128, ISPC, 2);

class mm_store_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                _mm_store_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_store_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_store_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_store_si128, ISPC, 2);

class mm_storeu_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storeu_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_storeu_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storeu_si128, ISPC, 2);

class mm_storel_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                _mm_storel_epi64((__m128i *)&Result[k], S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_storel_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_storel_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storel_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_storel_epi64, ISPC, 2);

class mm_maskmoveu_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int8[Iterations];
        Source1 = new int8[Iterations];
        Mask = new int8[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int8)k;
            Mask[k] = k & 1 ? -128 : 0; // -128 is 0x80 in two's complement
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                if (Mask[k] & 1 << 7) {
                    Result[k] = Source1[k];
                }
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 16) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i M = _mm_loadu_si128((__m128i *)&Mask[k]);
                _mm_maskmoveu_si128(S1, M, (char *)&Result[k]);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_maskmoveu_si128(Result, Source1, Mask, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k++) {
            if ((Mask[k] & 1 << 7) && Result[k] != Source1[k]) {
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

        delete[] Mask;
        Mask = nullptr;
    }

  private:
    int Iterations;

    int8 *Result;
    int8 *Source1;
    int8 *Mask;
};

BENCHMARK_CASE_POW2(mm_maskmoveu_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_maskmoveu_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_maskmoveu_si128, ISPC, 2);

class mm_move_epi64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128i R = _mm_move_epi64(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_move_epi64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_move_epi64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_epi64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_move_epi64, ISPC, 2);

#ifdef IS_X86_ARCH
class mm_stream_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load_pd(&Source1[k]);
                _mm_stream_pd(&Result[k], S1);
            }
            _mm_sfence();
            break;
        case 2:
            ispc::mm_stream_pd(Result, Source1, Iterations);
            _mm_sfence();
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

    double *Result;
    double *Source1;
};

#if AMD_PLATFORM == 0
BENCHMARK_CASE_POW2(mm_stream_pd, CPP, 0);
BENCHMARK_CASE_POW2(mm_stream_pd, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_stream_pd, ISPC, 2);
#endif
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_stream_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                _mm_stream_si128((__m128i *)&Result[k], S1);
            }
            _mm_sfence();
            break;
        case 2:
            ispc::mm_stream_si128(Result, Source1, Iterations);
            _mm_sfence();
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_stream_si128, CPP, 0);
BENCHMARK_CASE_POW2(mm_stream_si128, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_stream_si128, ISPC, 2);
#endif // IS_X86_ARCH

#ifdef IS_X86_ARCH
class mm_stream_si32 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k++) {
                Result[k] = Source1[k];
            }
            break;
        case 1:
            for (int k = 0; k < Iterations; k++) {
                _mm_stream_si32(&Result[k], Source1[k]);
            }
            _mm_sfence();
            break;
        case 2:
            ispc::mm_stream_si32(Result, Source1, Iterations);
            _mm_sfence();
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_stream_si32, CPP, 0);
BENCHMARK_CASE_POW2(mm_stream_si32, Intrinsic, 1);
BENCHMARK_CASE_POW2(mm_stream_si32, ISPC, 2);
#endif // IS_X86_ARCH

class mm_cvtsd_f64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_load_pd(&Source1[k]);
                Result[k] = _mm_cvtsd_f64(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsd_f64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0) {
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsd_f64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_f64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_f64, ISPC, 2);

class mm_castpd_ps : public TestBase {
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
            for (int k = 0; k < Iterations; k += 4) {
                const __m128d S1 = _mm_loadu_pd((double *)&Source1[k]);
                const __m128 R = _mm_castpd_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castpd_ps(Result, Source1, Iterations);
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

BENCHMARK_CASE_POW2(mm_castpd_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castpd_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castpd_ps, ISPC, 2);

class mm_castpd_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128d S1 = _mm_loadu_pd((double *)&Source1[k]);
                const __m128i R = _mm_castpd_si128(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castpd_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_castpd_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castpd_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castpd_si128, ISPC, 2);

class mm_castps_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128 S1 = _mm_loadu_ps((float *)&Source1[k]);
                const __m128d R = _mm_castps_pd(S1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castps_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_castps_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castps_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castps_pd, ISPC, 2);

class mm_castps_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int32[Iterations];
        Source1 = new int32[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (int32)k;
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
                const __m128 S1 = _mm_loadu_ps((float *)&Source1[k]);
                const __m128i R = _mm_castps_si128(S1);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castps_si128(Result, Source1, Iterations);
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

    int32 *Result;
    int32 *Source1;
};

BENCHMARK_CASE_POW2(mm_castps_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castps_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castps_si128, ISPC, 2);

class mm_castsi128_ps : public TestBase {
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
            for (int k = 0; k < Iterations; k += 4) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128 R = _mm_castsi128_ps(S1);
                _mm_storeu_ps(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castsi128_ps(Result, Source1, Iterations);
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

BENCHMARK_CASE_POW2(mm_castsi128_ps, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castsi128_ps, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castsi128_ps, ISPC, 2);

class mm_castsi128_pd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)k;
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
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i S1 = _mm_loadu_si128((__m128i *)&Source1[k]);
                const __m128d R = _mm_castsi128_pd(S1);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_castsi128_pd(Result, Source1, Iterations);
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

    double *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_castsi128_pd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castsi128_pd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_castsi128_pd, ISPC, 2);

class mm_cvtsd_si64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = (int64)nearbyint(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                Result[k] = _mm_cvtsd_si64(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsd_si64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != (int64)nearbyint(Source1[k]) && Result[k + 1] != 0) {
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

    int64 *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsd_si64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_si64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsd_si64, ISPC, 2);

class mm_cvttsd_si64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new double[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = (int64)(Source1[k]);
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                Result[k] = _mm_cvttsd_si64(S1);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvttsd_si64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != (int64)(Source1[k]) && Result[k + 1] != 0) {
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

    int64 *Result;
    double *Source1;
};

BENCHMARK_CASE_POW2(mm_cvttsd_si64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttsd_si64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvttsd_si64, ISPC, 2);

class mm_cvtsi64_sd : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new double[Iterations];
        Source1 = new double[Iterations];
        Source2 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 0;
            Source1[k] = (double)(k - Iterations / 2) * 0.1;
            Source2[k] = (int64)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = (double)(Source2[k]);
                Result[k + 1] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128d S1 = _mm_loadu_pd(&Source1[k]);
                const __m128d R = _mm_cvtsi64_sd(S1, Source2[k]);
                _mm_storeu_pd(&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi64_sd(Result, Source1, Source2, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != (double)Source2[k] && Result[k + 1] != Source2[k]) {
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

    double *Result;
    double *Source1;
    int64 *Source2;
};

BENCHMARK_CASE_POW2(mm_cvtsi64_sd, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi64_sd, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi64_sd, ISPC, 2);

class mm_cvtsi64_si128 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
                Result[k + 1] = 0;
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i R = _mm_cvtsi64_si128(Source1[k]);
                _mm_storeu_si128((__m128i *)&Result[k], R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi64_si128(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
            if (Result[k] != Source1[k] && Result[k + 1] != 0) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsi64_si128, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi64_si128, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi64_si128, ISPC, 2);

class mm_cvtsi128_si64 : public TestBase {
  public:
    virtual void SetUp(::benchmark::State &state) {
        Iterations = (int)state.range(0);

        Result = new int64[Iterations];
        Source1 = new int64[Iterations];

        for (int k = 0; k < Iterations; k++) {
            Result[k] = 2;
            Source1[k] = (int64)(k - Iterations / 2);
        }
    }

    virtual void Run(::benchmark::State &state, const unsigned int TestNumber) {
        switch (TestNumber) {
        case 0:
            for (int k = 0; k < Iterations; k += 2) {
                Result[k] = Source1[k];
            }
            break;
#ifdef IS_X86_ARCH
        case 1:
            for (int k = 0; k < Iterations; k += 2) {
                const __m128i R = _mm_loadu_si128((__m128i *)&Source1[k]);
                Result[k] = _mm_cvtsi128_si64(R);
            }
            break;
#endif // IS_X86_ARCH
        case 2:
            ispc::mm_cvtsi128_si64(Result, Source1, Iterations);
            break;
        }
    }

    virtual bool ResultsCorrect(const ::benchmark::State &state, const unsigned int TestNumber) {
        for (int k = 0; k < Iterations; k += 2) {
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

    int64 *Result;
    int64 *Source1;
};

BENCHMARK_CASE_POW2(mm_cvtsi128_si64, CPP, 0);
#ifdef IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi128_si64, Intrinsic, 1);
#endif // IS_X86_ARCH
BENCHMARK_CASE_POW2(mm_cvtsi128_si64, ISPC, 2);

// Main function
BENCHMARK_MAIN();
