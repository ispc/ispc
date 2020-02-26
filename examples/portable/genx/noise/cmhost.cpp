/*
  Copyright (c) 2019, Intel Corporation
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

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#undef min
#undef max
#endif // ISPC_IS_WINDOWS

#include "cm_rt_helpers.h"
#include "host_helpers.h"

#include <algorithm>

#define CORRECTNESS_THRESHOLD 0.0002
#define SZ 768 * 768
#define TIMEOUT (40 * 1000)

extern void noise_serial(float x0, float y0, float x1, float y1, int width, int height, float output[]);

using namespace hostutil;

// TODO: add scale factors
PageAlignedArray<float, SZ> out, gold;

static int run(int niter, int gx, int gy) {
    CmDevice *device = nullptr;
    CmProgram *program = nullptr;
    CmKernel *kernel = nullptr;

    for (int i = 0; i < SZ; i++)
        out.data[i] = -1;

    CMInitContext(device, kernel, program, "test.isa", "noise_ispc");

    // PARAMS
    unsigned int height = 768;
    unsigned int width = 768;

    float x0 = -10;
    float y0 = -10;
    float x1 = 10;
    float y1 = 10;

    kernel->SetKernelArg(0, sizeof(float), &x0);
    kernel->SetKernelArg(1, sizeof(float), &y0);
    kernel->SetKernelArg(2, sizeof(float), &x1);
    kernel->SetKernelArg(3, sizeof(float), &y1);
    kernel->SetKernelArg(4, sizeof(int), &width);
    kernel->SetKernelArg(5, sizeof(int), &height);

    void *buf_ref = out.data;

    CmBufferSVM *outSurf = nullptr;
    cm_result_check(
        device->CreateBufferSVM(height * width * sizeof(float), buf_ref, CM_SVM_ACCESS_FLAG_DEFAULT, outSurf));
    kernel->SetKernelArg(6, sizeof(void *), &buf_ref);

    // EXECUTION
    auto timings = execute(device, kernel, gx, gy, niter, false, TIMEOUT);
    timings.print(niter);

    void *res = out.data;
    cm_result_check(outSurf->GetAddress(res));

    cm_result_check(device->DestroyBufferSVM(outSurf));
    cm_result_check(device->DestroyKernel(kernel));
    cm_result_check(::DestroyCmDevice(device));

    // RESULT CHECK
    bool pass = true;
    if (niter == 1) {
        noise_serial(x0, y0, x1, y1, width, height, gold.data);
        double err = 0.0;
        double max_err = 0.0;

        int i = 0;
        for (; i < width * height; i++) {
            err = fabs(out.data[i] - gold.data[i]);
            max_err = std::max(err, max_err);
            if (err > CORRECTNESS_THRESHOLD)
                pass = false;
        }
        if (!pass) {
            std::cout << "Correctness test failed on " << i << "th value." << std::endl;
            std::cout << "Was " << out.data[i] << ", should be " << gold.data[i] << std::endl;
        } else {
            std::cout << "Passed!"
                      << " Max error:" << max_err << std::endl;
        }
    }

    return (pass) ? 0 : 1;
}

int main(int argc, char *argv[]) {
    int niterations = 1;
    int gx = 1, gy = 1;
    niterations = atoi(argv[1]);
    if (argc == 4) {
        gx = atoi(argv[2]);
        gy = atoi(argv[3]);
    }

    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;
    success = run(niterations, gx, gy);

    return success;
}
