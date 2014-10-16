// -*- mode: c++ -*-
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

#include "options_defs.h"

// Cumulative normal distribution function
static inline float
CND(float X) {
    float L = abs(X);

    float k = 1.0 / (1.0 + 0.2316419 * L);
    float k2 = k*k;
    float k3 = k2*k;
    float k4 = k2*k2;
    float k5 = k3*k2;

    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * exp(-L * L * .5f);

    if (X > 0.f)
        w = 1.0 - w;
    return w;
}

task void
bs_task(uniform float Sa[], uniform float Xa[], uniform float Ta[],
        uniform float ra[], uniform float va[],
        uniform float result[], uniform int count) {
    uniform int first = taskIndex * (count/taskCount);
    uniform int last = min(count, (int)((taskIndex+1) * (count/taskCount)));

    foreach (i = first ... last) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];

        float d1 = (log(S/X) + (r + v * v * .5f) * T) / (v * sqrt(T));
        float d2 = d1 - v * sqrt(T);

        result[i] = S * CND(d1) - X * exp(-r * T) * CND(d2);
    }
}

export void
black_scholes_ispc_tasks(uniform float Sa[], uniform float Xa[], uniform float Ta[],
                         uniform float ra[], uniform float va[],
                         uniform float result[], uniform int count) {
    uniform int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
    launch[nTasks] bs_task(Sa, Xa, Ta, ra, va, result, count);
}

/********/


export void
black_scholes_ispc(uniform float Sa[], uniform float Xa[], uniform float Ta[],
                   uniform float ra[], uniform float va[],
                   uniform float result[], uniform int count) {
    foreach (i = 0 ... count) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];

        float d1 = (log(S/X) + (r + v * v * .5f) * T) / (v * sqrt(T));
        float d2 = d1 - v * sqrt(T);

        result[i] = S * CND(d1) - X * exp(-r * T) * CND(d2);
    }
}


static inline float
binomial_put(float S, float X, float T, float r, float v) {
    float V[BINOMIAL_NUM];

    float dt = T / BINOMIAL_NUM;
    float u = exp(v * sqrt(dt));
    float d = 1. / u;
    float disc = exp(r * dt);
    float Pu = (disc - d) / (u - d);

#ifndef __NVPTX__

    for (uniform int j = 0; j < BINOMIAL_NUM; ++j) {
        float upow = pow(u, (float)(2*j-BINOMIAL_NUM));
        V[j] = max(0., X - S * upow);
    }
    for (uniform int j = BINOMIAL_NUM-1; j >= 0; --j)
        for (uniform int k = 0; k < j; ++k)
            V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;

#else

    /* loop unrolling helps NVVM to place V -> registers therefore boosting performance */
    /* takes looong time to compile... */
#if BINOMIAL_NUM != 64
#error "Cannot unroll. Please use generic version above"
#endif

    // with PTX target unroll loops which will store data in registers..

    /* first loop */

#define OP(j) { \
        float upow = pow(u, (float)(2*(j)-BINOMIAL_NUM)); \
        V[j] = max(0., X - S * upow); }
#define OP10(k) \
    OP(k+0); OP(k+1); OP(k+2); OP(k+3); OP(k+4) \
    OP(k+5); OP(k+6); OP(k+7); OP(k+8); OP(k+9);
    OP10(0)
    OP10(10)
    OP10(20)
    OP10(30)
    OP10(40)
    OP10(50)
    OP(60)
    OP(61)
    OP(62)
    OP(63)
#undef OP10
#undef OP

    /* second loop */

#define OP(j) {\
  for (uniform int k = 0; k < (j); ++k) \
      V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc; }
#define OP10(k) \
  OP(k+9); OP(k+8); OP(k+7); OP(k+6); OP(k+5); \
  OP(k+4); OP(k+3); OP(k+2); OP(k+1); OP(k+0);
  OP(63)
  OP(62)
  OP(61)
  OP(60)
  OP10(50)
  OP10(40)
  OP10(30)
  OP10(20)
  OP10(10)
  OP10(0)
#undef OP10
#undef OP

#endif
    return V[0];
}


export void
binomial_put_ispc(uniform float Sa[], uniform float Xa[], uniform float Ta[],
                  uniform float ra[], uniform float va[],
                  uniform float result[], uniform int count) {
    foreach (i = 0 ... count) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        result[i] = binomial_put(S, X, T, r, v);
    }
}


task void
binomial_task(uniform float Sa[], uniform float Xa[],
              uniform float Ta[], uniform float ra[],
              uniform float va[], uniform float result[],
              uniform int count) {
    uniform int first = taskIndex * (count/taskCount);
    uniform int last = min(count, (int)((taskIndex+1) * (count/taskCount)));

    foreach (i = first ... last) {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        result[i] = binomial_put(S, X, T, r, v);
    }
}


export void
binomial_put_ispc_tasks(uniform float Sa[], uniform float Xa[],
                        uniform float Ta[], uniform float ra[],
                        uniform float va[], uniform float result[],
                        uniform int count) {
    uniform int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
    launch[nTasks] binomial_task(Sa, Xa, Ta, ra, va, result, count);
}
