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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <algorithm>
using std::max;

#include "options_defs.h"
#include "../timing.h"
#include "../cpuid.h"

#include "options_ispc.h"
using namespace ispc;

extern void black_scholes_serial(float Sa[], float Xa[], float Ta[], 
                                 float ra[], float va[], 
                                 float result[], int count);

extern void binomial_put_serial(float Sa[], float Xa[], float Ta[], 
                                float ra[], float va[], 
                                float result[], int count);

// Make sure that the vector ISA used during compilation is supported by
// the processor.  The ISPC_TARGET_* macro is set in the ispc-generated
// header file that we include above.
static void
ensureTargetISAIsSupported() {
#if defined(ISPC_TARGET_SSE2)
    bool isaSupported = CPUSupportsSSE2();
    const char *target = "SSE2";
#elif defined(ISPC_TARGET_SSE4)
    bool isaSupported = CPUSupportsSSE4();
    const char *target = "SSE4";
#elif defined(ISPC_TARGET_AVX)
    bool isaSupported = CPUSupportsAVX();
    const char *target = "AVX";
#else
#error "Unknown ISPC_TARGET_* value"
#endif
    if (!isaSupported) {
        fprintf(stderr, "***\n*** Error: the ispc-compiled code uses the %s instruction "
                "set, which isn't\n***        supported by this computer's CPU!\n", target);
        fprintf(stderr, "***\n***        Please modify the "
#ifdef _MSC_VER
                "MSVC project file "
#else
                "Makefile "
#endif
                "to select another target (e.g. sse2)\n***\n");
        exit(1);
    }
}


int main() {
    ensureTargetISAIsSupported();
    
    float *S = new float[N_OPTIONS];
    float *X = new float[N_OPTIONS];
    float *T = new float[N_OPTIONS];
    float *r = new float[N_OPTIONS];
    float *v = new float[N_OPTIONS];
    float *result = new float[N_OPTIONS];

    for (int i = 0; i < N_OPTIONS; ++i) {
        S[i] = 100;  // stock price
        X[i] = 98;   // option strike price
        T[i] = 2;    // time (years)
        r[i] = .02;  // risk-free interest rate
        v[i] = 5;    // volatility
    }

    //
    // Binomial options pricing model, ispc implementation
    //
    reset_and_start_timer();
    binomial_put_ispc(S, X, T, r, v, result, N_OPTIONS);
    double binomial_ispc = get_elapsed_mcycles();
    float sum = 0.f;
    for (int i = 0; i < N_OPTIONS; ++i)
        sum += result[i];
    printf("[binomial ispc]:\t\t[%.3f] million cycles (avg %f)\n", 
           binomial_ispc, sum / N_OPTIONS);

    //
    // Binomial options, serial implementation
    //
    reset_and_start_timer();
    binomial_put_serial(S, X, T, r, v, result, N_OPTIONS);
    double binomial_serial = get_elapsed_mcycles();
    sum = 0.f;
    for (int i = 0; i < N_OPTIONS; ++i)
        sum += result[i];
    printf("[binomial serial]:\t\t[%.3f] million cycles (avg %f)\n", 
           binomial_serial, sum / N_OPTIONS);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", binomial_serial / binomial_ispc);

    //
    // Black-Scholes options pricing model, ispc implementation
    //
    sum = 0.f;
    reset_and_start_timer();
    for (int a = 0; a < N_BLACK_SCHOLES_ROUNDS; ++a) {
        black_scholes_ispc(S, X, T, r, v, result, N_OPTIONS);
        for (int i = 0; i < N_OPTIONS; ++i)
            sum += result[i];
    }
    double bs_ispc = get_elapsed_mcycles();
    printf("[black-scholes ispc]:\t\t[%.3f] million cycles (avg %f)\n", 
           bs_ispc, sum / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS));

    //
    // Black-Scholes options pricing model, serial implementation
    //
    sum = 0.f;
    reset_and_start_timer();
    for (int a = 0; a < N_BLACK_SCHOLES_ROUNDS; ++a) {
        black_scholes_serial(S, X, T, r, v, result, N_OPTIONS);
        for (int i = 0; i < N_OPTIONS; ++i)
            sum += result[i];
    }
    double bs_serial = get_elapsed_mcycles();
    printf("[black-scholes serial]:\t\t[%.3f] million cycles (avg %f)\n", bs_serial, 
           sum / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS));

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", bs_serial / bs_ispc);

    return 0;
}
