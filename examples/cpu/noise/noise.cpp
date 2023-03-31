/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

#include "../../common/timing.h"
#include "noise_ispc.h"
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
using namespace ispc;

extern void noise_serial(float x0, float y0, float x1, float y1, int width, int height, float output[]);

/* Write a PPM image file with the image */
static void writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    if (!fp) {
        printf("Couldn't open a file '%s'\n", fn);
        exit(-1);
    }
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0)
            v = 0;
        if (v > 255)
            v = 255;
        for (int j = 0; j < 3; ++j)
            fputc((char)v, fp);
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 1};
    unsigned int width = 768;
    unsigned int height = 768;
    float x0 = -10;
    float x1 = 10;
    float y0 = -10;
    float y1 = 10;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            width *= scale;
            height *= scale;
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }
    float *buf = new float[width * height];

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        noise_ispc(x0, y0, x1, y1, width, height, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", dt);
        minISPC = std::min(minISPC, dt);
    }

    printf("[noise ispc]:\t\t\t[%.3f] million cycles\n", minISPC);
    writePPM(buf, width, height, "noise-ispc.ppm");

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
        buf[i] = 0;

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        noise_serial(x0, y0, x1, y1, width, height, buf);
        double dt = get_elapsed_mcycles();
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
        minSerial = std::min(minSerial, dt);
    }

    printf("[noise serial]:\t\t\t[%.3f] million cycles\n", minSerial);
    writePPM(buf, width, height, "noise-serial.ppm");

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial / minISPC);

    return 0;
}
