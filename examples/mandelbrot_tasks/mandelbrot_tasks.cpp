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

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string.h>
#include "../timing.h"
#include "mandelbrot_tasks_ispc.h"
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
    printf("Wrote image file %s\n", fn);
}


static void usage() {
    fprintf(stderr, "usage: mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {7, 1};
    unsigned int width = 1536;
    unsigned int height = 1024;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            if (scale == 0.f)
                usage();
            width *= scale;
            height *= scale;
            // round up to multiples of 16
            width = (width + 0xf) & ~0xf;
            height = (height + 0xf) & ~0xf;
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }


    int maxIterations = 512;
    int *buf = new int[width*height];

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        // Clear out the buffer
        for (unsigned int i = 0; i < width * height; ++i)
            buf[i] = 0;
        reset_and_start_timer();
        mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", dt);
        minISPC = std::min(minISPC, dt);
    }

    printf("[mandelbrot ispc+tasks]:\t[%.3f] million cycles\n", minISPC);
    writePPM(buf, width, height, "mandelbrot-ispc.ppm");


    // 
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        // Clear out the buffer
        for (unsigned int i = 0; i < width * height; ++i)
            buf[i] = 0;
        reset_and_start_timer();
        mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
        minSerial = std::min(minSerial, dt);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] million cycles\n", minSerial);
    writePPM(buf, width, height, "mandelbrot-serial.ppm");

    printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", minSerial/minISPC);

    return 0;
}
