/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#define WINDOWS
#endif

#ifdef WINDOWS
#define CALLINGCONV /*__vectorcall*/
#else
#define CALLINGCONV
#endif

void CALLINGCONV xyzSumAOS(float *a, int count, float *zeros, float *result) {
    float xsum = 0, ysum = 0, zsum = 0;
    for (int i = 0; i < count; i += 3) {
        xsum += a[i];
        ysum += a[i + 1];
        zsum += a[i + 2];
    }
    result[0] = xsum;
    result[1] = ysum;
    result[2] = zsum;
}

void CALLINGCONV xyzSumSOA(float *a, int count, float *zeros, float *result) {
    float xsum = 0, ysum = 0, zsum = 0;
    for (int i = 0; i < count / 3; ++i) {
        float *p = a + (i >> 3) * 24 + (i & 7);
        xsum += p[0];
        ysum += p[8];
        zsum += p[16];
    }
    result[0] = xsum;
    result[1] = ysum;
    result[2] = zsum;
}
