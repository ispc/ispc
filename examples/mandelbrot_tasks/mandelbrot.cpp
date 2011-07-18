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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning (disable: 4244)
#pragma warning (disable: 4305)
#endif

#include <stdio.h>
#include <algorithm>
#include "../timing.h"
#include "../cpuid.h"
#include "mandelbrot_ispc.h"
using namespace ispc;

extern void mandelbrot_serial(float x0, float y0, float x1, float y1,
                              int width, int height, int maxIterations,
                              int output[]);

/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
}


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
    unsigned int width = 1536;
    unsigned int height = 1024;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    ensureTargetISAIsSupported();

    extern void TasksInit();
    TasksInit();

    int maxIterations = 512;
    int *buf = new int[width*height];

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        minISPC = std::min(minISPC, dt);
    }

    printf("[mandelbrot ispc+tasks]:\t[%.3f] million cycles\n", minISPC);
    writePPM(buf, width, height, "mandelbrot-ispc.ppm");

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
        buf[i] = 0;

    // 
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        minSerial = std::min(minSerial, dt);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] millon cycles\n", minSerial);
    writePPM(buf, width, height, "mandelbrot-serial.ppm");

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);

    return 0;
}
