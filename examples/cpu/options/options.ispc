// -*- mode: c++ -*-
/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
    uniform int last = min(count, (uniform int)((taskIndex+1) * (count/taskCount)));

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
    uniform int nTasks = max((uniform int)64, (uniform int)count/16384);
    launch[nTasks] bs_task(Sa, Xa, Ta, ra, va, result, count);
}


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

    for (uniform int j = 0; j < BINOMIAL_NUM; ++j) {
        float upow = pow(u, (uniform float)(2*j-BINOMIAL_NUM));
        V[j] = max(0., X - S * upow);
    }

    for (uniform int j = BINOMIAL_NUM-1; j >= 0; --j)
        for (uniform int k = 0; k < j; ++k)
            V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;
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
    uniform int last = min(count, (uniform int)((taskIndex+1) * (count/taskCount)));

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
    uniform int nTasks = max((uniform int)64, (uniform int)count/16384);
    launch[nTasks] binomial_task(Sa, Xa, Ta, ra, va, result, count);
}
