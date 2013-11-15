// -*- mode: c++ -*-
/*
  Copyright (c) 2010-2011, Intel Corporation
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

#define programCount 32
#define programIndex (threadIdx.x & 31)
#define taskIndex (blockIdx.x*4 + (threadIdx.x >> 5))
#define taskCount (gridDim.x*4)
#define warpIdx (threadIdx.x >> 5)

// Cumulative normal distribution function
//
__device__
static inline float
CND(float X) {
    float L = fabsf(X);

    float k = 1.0f / (1.0f + 0.2316419f * L);
    float k2 = k*k;
    float k3 = k2*k;
    float k4 = k2*k2;
    float k5 = k3*k2;

    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * expf(-L * L * .5f);

    if (X > 0.f)
        w = 1.0f - w;
    return w;
}

__global__ 
void bs_task( float Sa[],  float Xa[],  float Ta[],
    float ra[],  float va[], 
    float result[],  int count) {
     int first = taskIndex * (count/taskCount);
     int last = min(count, (int)((taskIndex+1) * (count/taskCount)));

    for (int i = programIndex + first; i < last; i += programCount)
      if (i < last)
    {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];

        float d1 = (logf(S/X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);

        result[i] = S * CND(d1) - X * expf(-r * T) * CND(d2);
    }
}

extern "C"
__global__ void
black_scholes_ispc_tasks( float Sa[],  float Xa[],  float Ta[],
                          float ra[],  float va[], 
                          float result[],  int count) {
     int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
    bs_task<<<nTasks/4,128>>>(Sa, Xa, Ta, ra, va, result, count);
}

/********/


__device__
static inline float
binomial_put(float S, float X, float T, float r, float v) {
#if 0
    float V[BINOMIAL_NUM];
#else
    __shared__ float VSH[BINOMIAL_NUM*4];
    float *V = VSH + warpIdx*BINOMIAL_NUM;
#endif

    float dt = T / BINOMIAL_NUM;
    float u = exp(v * sqrt(dt));
    float d = 1.f / u;
    float disc = exp(r * dt);
    float Pu = (disc - d) / (u - d);

#pragma unroll
    for ( int j = 0; j < BINOMIAL_NUM; ++j) {
        float upow = pow(u, (float)(2*j-BINOMIAL_NUM));
        V[j] = max(0., X - S * upow);
    }

#pragma unroll
    for ( int j = BINOMIAL_NUM-1; j >= 0; --j)
#pragma unroll
        for ( int k = 0; k < j; ++k)
            V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;
    return V[0];
}



__global__ void
binomial_task( float Sa[],  float Xa[], 
               float Ta[],  float ra[], 
               float va[],  float result[], 
               int count) {
     int first = taskIndex * (count/taskCount);
     int last = min(count, (int)((taskIndex+1) * (count/taskCount)));

    for (int i = programIndex + first; i < last; i += programCount)
      if (i < last)
      {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        result[i] = binomial_put(S, X, T, r, v);
    }
}


extern "C" __global__ void
binomial_put_ispc_tasks( float Sa[],  float Xa[], 
                         float Ta[],  float ra[], 
                         float va[],  float result[], 
                         int count) {
  int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
    if (programIndex == 0)
      binomial_task<<<nTasks/4,128>>>(Sa, Xa, Ta, ra, va, result, count);
    cudaDeviceSynchronize();
}
