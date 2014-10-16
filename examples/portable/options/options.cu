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
#include "cuda_helpers.cuh"

__device__ static inline void __range_reduce_log(float input, float * reduced,
                                      int * exponent) {
    int int_version = __float_as_int(input); //intbits(input);
    // single precision = SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM
    // exponent mask    = 0111 1111 1000 0000 0000 0000 0000 0000
    //                    0x7  0xF  0x8  0x0  0x0  0x0  0x0  0x0
    // non-exponent     = 1000 0000 0111 1111 1111 1111 1111 1111
    //                  = 0x8  0x0  0x7  0xF  0xF  0xF  0xF  0xF

    //const int exponent_mask(0x7F800000)
    const int nonexponent_mask = 0x807FFFFF;

    // We want the reduced version to have an exponent of -1 which is -1 + 127 after biasing or 126
    const int exponent_neg1 = (126l << 23);
    // NOTE(boulos): We don't need to mask anything out since we know
    // the sign bit has to be 0. If it's 1, we need to return infinity/nan
    // anyway (log(x), x = +-0 -> infinity, x < 0 -> NaN).
    int biased_exponent = int_version >> 23; // This number is [0, 255] but it means [-127, 128]

    int offset_exponent = biased_exponent + 1; // Treat the number as if it were 2^{e+1} * (1.m)/2
    *exponent = offset_exponent - 127; // get the real value

    // Blend the offset_exponent with the original input (do this in
    // int for now, until I decide if float can have & and &not)
    int blended = (int_version & nonexponent_mask) | (exponent_neg1);
    *reduced = __int_as_float(blended); //floatbits(blended);
}


__device__ static inline float __Logf(const float x_full)
{
#if 1
  return __logf(x_full);
#else
  float reduced;
  int exponent;

  const int NaN_bits = 0x7fc00000;
  const int Neg_Inf_bits = 0xFF800000;
  const float NaN = __int_as_float(NaN_bits); //floatbits(NaN_bits);
  const float neg_inf = __int_as_float(Neg_Inf_bits); //floatbits(Neg_Inf_bits);
  bool use_nan = x_full < 0.f;
  bool use_inf = x_full == 0.f;
  bool exceptional = use_nan || use_inf;
  const float one = 1.0f;

  float patched = exceptional ? one : x_full;
  __range_reduce_log(patched, &reduced, &exponent);

  const float ln2 = 0.693147182464599609375f;

  float x1 = one - reduced;
  const float c1 = 0.50000095367431640625f;
  const float c2 = 0.33326041698455810546875f;
  const float c3 = 0.2519190013408660888671875f;
  const float c4 = 0.17541764676570892333984375f;
  const float c5 = 0.3424419462680816650390625f;
  const float c6 = -0.599632322788238525390625f;
  const float c7 = +1.98442304134368896484375f;
  const float c8 = -2.4899270534515380859375f;
  const float c9 = +1.7491014003753662109375f;

  float result = x1 * c9 + c8;
  result = x1 * result + c7;
  result = x1 * result + c6;
  result = x1 * result + c5;
  result = x1 * result + c4;
  result = x1 * result + c3;
  result = x1 * result + c2;
  result = x1 * result + c1;
  result = x1 * result + one;

  // Equation was for -(ln(red)/(1-red))
  result *= -x1;
  result += (float)(exponent) * ln2;

  return exceptional ? (use_nan ? NaN : neg_inf) : result;
#endif
}

__device__ static inline float __Expf(const float x_full)
{
#if 1
  return __expf(x_full);
#else
  const float ln2_part1 = 0.6931457519f;
  const float ln2_part2 = 1.4286067653e-6f;
  const float one_over_ln2 = 1.44269502162933349609375f;

  float scaled = x_full * one_over_ln2;
  float k_real = floor(scaled);
  int k = (int)k_real;

  // Reduced range version of x
  float x = x_full - k_real * ln2_part1;
  x -= k_real * ln2_part2;

  // These coefficients are for e^x in [0, ln(2)]
  const float one = 1.f;
  const float c2 = 0.4999999105930328369140625f;
  const float c3 = 0.166668415069580078125f;
  const float c4 = 4.16539050638675689697265625e-2f;
  const float c5 = 8.378830738365650177001953125e-3f;
  const float c6 = 1.304379315115511417388916015625e-3f;
  const float c7 = 2.7555381529964506626129150390625e-4f;

  float result = x * c7 + c6;
  result = x * result + c5;
  result = x * result + c4;
  result = x * result + c3;
  result = x * result + c2;
  result = x * result + one;
  result = x * result + one;

  // Compute 2^k (should differ for float and double, but I'll avoid
  // it for now and just do floats)
  const int fpbias = 127;
  int biased_n = k + fpbias;
  bool overflow = k > fpbias;
  // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
  // we've got underflow. -127 * ln(2) -> -88.02. So the most
  // negative float input that doesn't result in zero is like -88.
  bool underflow = (biased_n <= 0);
  const int InfBits = 0x7f800000;
  biased_n <<= 23;
  // Reinterpret this thing as float
  float two_to_the_n = __int_as_float(biased_n); //floatbits(biased_n);
  // Handle both doubles and floats (hopefully eliding the copy for float)
  float elemtype_2n = two_to_the_n;
  result *= elemtype_2n;
//  result = overflow ? floatbits(InfBits) : result;
  result = overflow ? __int_as_float(InfBits) : result;
  result = underflow ? 0.0f : result;
  return result;
#endif
}

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
    w *= invSqrt2Pi * __Expf(-L * L * .5f);

    if (X > 0.f)
        w = 1.0f - w;
    return w;
}

__global__
void bs_task( float Sa[],  float Xa[],  float Ta[],
    float ra[],  float va[],
    float result[],  int count) {
  if (taskIndex >= taskCount) return;
     int first = taskIndex * (count/taskCount);
     int last = min(count, (int)((taskIndex+1) * (count/taskCount)));

    for (int i = programIndex + first; i < last; i += programCount)
      if (i < last)
    {
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];

        float d1 = (__Logf(S/X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);

        result[i] = S * CND(d1) - X * __Expf(-r * T) * CND(d2);
    }
}

extern "C"
__global__ void
black_scholes_ispc_tasks___export( float Sa[],  float Xa[],  float Ta[],
                          float ra[],  float va[],
                          float result[],  int count) {
  int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
  launch(nTasks,1,1,bs_task)
    (Sa, Xa, Ta, ra, va, result, count);
  cudaDeviceSynchronize();
}
extern "C"
__host__ void
black_scholes_ispc_tasks( float Sa[],  float Xa[],  float Ta[],
                          float ra[],  float va[],
                          float result[],  int count) {
  black_scholes_ispc_tasks___export<<<1,32>>>(Sa,Xa,Ta,ra,va,result,count);
  cudaDeviceSynchronize();
}

/********/


template<int NBEG, int NEND, int STEP>
struct loop
{
  __device__ static void op1(float V[], const float u, const float X, const float S)
  {
    const int j = NBEG;
    float upow = powf(u, (float)(2*j-BINOMIAL_NUM));
    V[j] = max(0.0f, X - S * upow);
    loop<j+STEP,NEND,STEP>::op1(V,u,X,S);
  }
  __device__ static void op2(float V[], const float Pu, const float disc)
  {
    const int j = NBEG;
#pragma unroll
    for ( int k = 0; k < j; ++k)
      V[k] = ((1.0f - Pu) * V[k] + Pu * V[k+ 1]) / disc;
    loop<j+STEP,NEND,STEP>::op2(V, Pu,disc);
  }
};

template<int NEND, int STEP>
struct loop<NEND,NEND,STEP>
{
  __device__ static void op1(float V[], const float u, const float X, const float S) {}
  __device__ static void op2(float V[], const float Pu, const float disc) {}
};

__device__
static inline float
binomial_put(float S, float X, float T, float r, float v)
{

  float V[BINOMIAL_NUM];

  float dt = T / BINOMIAL_NUM;
  float u = exp(v * sqrt(dt));
  float d = 1.f / u;
  float disc = exp(r * dt);
  float Pu = (disc - d) / (u - d);

#if 0  /* slow */
  for ( int j = 0; j < BINOMIAL_NUM; ++j) {
    float upow = powf(u, (float)(2*j-BINOMIAL_NUM));
    V[j] = max(0.0f, X - S * upow);
  }
  for ( int j = BINOMIAL_NUM-1; j >= 0; --j)
    for ( int k = 0; k < j; ++k)
      V[k] = ((1.0f - Pu) * V[k] + Pu * V[k+ 1]) / disc;
#else  /* with loop unrolling, stores resutls in registers */
  loop<0,BINOMIAL_NUM,1>::op1(V,u,X,S);
  loop<BINOMIAL_NUM-1, -1, -1>::op2(V, Pu, disc);
#endif
  return V[0];
}



__global__ void
binomial_task( float Sa[],  float Xa[],
               float Ta[],  float ra[],
               float va[],  float result[],
               int count)
{
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
binomial_put_ispc_tasks___export( float Sa[],  float Xa[],
                         float Ta[],  float ra[],
                         float va[],  float result[],
                         int count) {
  int nTasks = 2048; //count/16384; //max((int)64, (int)count/16384);
  launch(nTasks,1,1,binomial_task)
    (Sa, Xa, Ta, ra, va, result, count);
  cudaDeviceSynchronize();
}
extern "C"
__host__ void
binomial_put_ispc_tasks( float Sa[],  float Xa[],  float Ta[],
                          float ra[],  float va[],
                          float result[],  int count) {

  cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
  binomial_put_ispc_tasks___export<<<1,32>>>(Sa,Xa,Ta,ra,va,result,count);
  cudaDeviceSynchronize();
}
