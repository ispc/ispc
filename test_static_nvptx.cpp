/*
  Copyright (c) 2010-2014, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#endif // ISPC_IS_WINDOWS

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdint.h>
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif

#include "ispc_malloc.h"

#define N 32
extern "C" {
int width() { return N; }
extern void f_v(float *result);
extern void f_f(float *result, float *a);
extern void f_fu(float *result, float *a, float b);
extern void f_fi(float *result, float *a, int *b);
extern void f_du(float *result, double *a, double b);
extern void f_duf(float *result, double *a, float b);
extern void f_di(float *result, double *a, int *b);
extern void result(float *val);
}

int main(int argc, char *argv[]) {
    int w = width();
    assert(w <= N);

    float *returned_result = new float[N * 4];
    float *vfloat = new float[N * 4];
    double *vdouble = new double[N * 4];
    int *vint = new int[N * 4];
    int *vint2 = new int[N * 4];

    for (int i = 0; i < N * 4; ++i) {
        returned_result[i] = -1e20;
        vfloat[i] = i + 1;
        vdouble[i] = i + 1;
        vint[i] = 2 * (i + 1);
        vint2[i] = i + 5;
    }

    float b = 5.;

#if (TEST_SIG == 0)
    f_v(returned_result);
#elif (TEST_SIG == 1)
    f_f(returned_result, vfloat);
#elif (TEST_SIG == 2)
    f_fu(returned_result, vfloat, b);
#elif (TEST_SIG == 3)
    f_fi(returned_result, vfloat, vint);
#elif (TEST_SIG == 4)
    f_du(returned_result, vdouble, 5.);
#elif (TEST_SIG == 5)
    f_duf(returned_result, vdouble, 5.f);
#elif (TEST_SIG == 6)
    f_di(returned_result, vdouble, vint2);
#else
#error "Unknown or unset TEST_SIG value"
#endif

    float *expected_result = new float[N];
    memset(expected_result, 0, N * sizeof(float));
    result(expected_result);

    int errors = 0;
    for (int i = 0; i < w; ++i) {
        if (returned_result[i] != expected_result[i]) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n", argv[0], i, returned_result[i],
                   returned_result[i], expected_result[i], expected_result[i]);
            ++errors;
#endif // EXPECT_FAILURE
        }
    }

#ifdef EXPECT_FAILURE
    // Don't expect to get here
    return 0;
#else
    return errors > 0;
#endif
}
