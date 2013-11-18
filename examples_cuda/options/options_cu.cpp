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

#define NOMINMAX

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <algorithm>
using std::max;

#include "options_defs.h"
#include "../timing.h"

#include "options_ispc.h"
using namespace ispc;
#include <sys/time.h>
#include "../cuda_ispc.h"

static void usage() {
    printf("usage: options [--count=<num options>]\n");
}


int main(int argc, char *argv[]) {
    int nOptions = 128*1024;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--count=", 8) == 0) {
            nOptions = atoi(argv[i] + 8);
            if (nOptions <= 0) {
                usage();
                exit(1);
            }
        }
    }

    float *S = new float[nOptions];
    float *X = new float[nOptions];
    float *T = new float[nOptions];
    float *r = new float[nOptions];
    float *v = new float[nOptions];
    float *result = new float[nOptions];

    for (int i = 0; i < nOptions; ++i) {
        S[i] = 100;  // stock price
        X[i] = 98;   // option strike price
        T[i] = 2;    // time (years)
        r[i] = .02;  // risk-free interest rate
        v[i] = 5;    // volatility
    }

    /*******************/
    createContext();
    /*******************/
    devicePtr d_S = deviceMalloc(nOptions*sizeof(float));
    devicePtr d_X = deviceMalloc(nOptions*sizeof(float));
    devicePtr d_T = deviceMalloc(nOptions*sizeof(float));
    devicePtr d_r = deviceMalloc(nOptions*sizeof(float));
    devicePtr d_v = deviceMalloc(nOptions*sizeof(float));
    devicePtr d_result = deviceMalloc(nOptions*sizeof(float));

    memcpyH2D(d_S, S, nOptions*sizeof(float));
    memcpyH2D(d_X, X, nOptions*sizeof(float));
    memcpyH2D(d_T, T, nOptions*sizeof(float));
    memcpyH2D(d_r, r, nOptions*sizeof(float));
    memcpyH2D(d_v, v, nOptions*sizeof(float));

    double sum;

    //
    // Binomial options pricing model, ispc implementation
    //
    const bool print_log = false;
    const int nreg = 64;
    double binomial_ispc = 1e30;
#if 0
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        const double t0 = rtc();
        const char * func_name = "binomial_put_ispc";
        void *func_args[] = {&d_S, &d_X, &d_T, &d_r, &d_v, &d_result, &nOptions};
        double dt = CUDALaunch(NULL, func_name, func_args, print_log, nreg);
        dt *= 1e3;
        memcpyD2H(result, d_result, nOptions*sizeof(float));
        sum = 0.;
        for (int i = 0; i < nOptions; ++i)
            sum += result[i];
        binomial_ispc = std::min(binomial_ispc, dt);
    }
    printf("[binomial ispc, 1 thread]:\t[%.3f] million cycles (avg %f)\n", 
           binomial_ispc, sum / nOptions);
    for (int i = 0; i < nOptions; ++i)
      result[i] = 0.0;
    memcpyH2D(d_result, result, nOptions*sizeof(float));
#endif

    //
    // Binomial options pricing model, ispc implementation, tasks
    //
    double binomial_tasks = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        const char * func_name = "binomial_put_ispc_tasks";
        void *func_args[] = {&d_S, &d_X, &d_T, &d_r, &d_v, &d_result, &nOptions};
        double dt = CUDALaunch(NULL, func_name, func_args, print_log, nreg);
        dt *= 1e3;
        memcpyD2H(result, d_result, nOptions*sizeof(float));
        sum = 0.;
        for (int i = 0; i < nOptions; ++i)
            sum += result[i];
        binomial_tasks = std::min(binomial_tasks, dt);
    }
    printf("[binomial ispc, tasks]:\t\t[%.3f] million cycles (avg %f)\n", 
           binomial_tasks, sum / nOptions);
    for (int i = 0; i < nOptions; ++i)
      result[i] = 0.0;
    memcpyH2D(d_result, result, nOptions*sizeof(float));


    //
    // Black-Scholes options pricing model, ispc implementation, 1 thread
    //
    double bs_ispc = 1e30;
#if 0
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        const char * func_name = "black_scholes_ispc";
        void *func_args[] = {&d_S, &d_X, &d_T, &d_r, &d_v, &d_result, &nOptions};
        double dt = CUDALaunch(NULL, func_name, func_args, print_log, nreg);
        dt *= 1e3;
        memcpyD2H(result, d_result, nOptions*sizeof(float));
        sum = 0.;
        for (int i = 0; i < nOptions; ++i)
            sum += result[i];
        bs_ispc = std::min(bs_ispc, dt);
    }
    printf("[black-scholes ispc, 1 thread]:\t[%.3f] million cycles (avg %f)\n", 
           bs_ispc, sum / nOptions);
    for (int i = 0; i < nOptions; ++i)
      result[i] = 0.0;
    memcpyH2D(d_result, result, nOptions*sizeof(float));
#endif

    //
    // Black-Scholes options pricing model, ispc implementation, tasks
    //
    double bs_ispc_tasks = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        const char * func_name = "black_scholes_ispc_tasks";
        void *func_args[] = {&d_S, &d_X, &d_T, &d_r, &d_v, &d_result, &nOptions};
        double dt = CUDALaunch(NULL, func_name, func_args, print_log, nreg);
        dt *= 1e3;
        memcpyD2H(result, d_result, nOptions*sizeof(float));
        sum = 0.;
        for (int i = 0; i < nOptions; ++i)
            sum += result[i];
        bs_ispc_tasks = std::min(bs_ispc_tasks, dt);
    for (int i = 0; i < nOptions; ++i)
      result[i] = 0.0;
    memcpyH2D(d_result, result, nOptions*sizeof(float));
    }
    printf("[black-scholes ispc, tasks]:\t[%.3f] million cycles (avg %f)\n", 
           bs_ispc_tasks, sum / nOptions);


    return 0;
}
