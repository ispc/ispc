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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include <algorithm>
#include <map>
#include <math.h>
#include <string>
#include <sys/types.h>

#include "ao_ispc.h"
using namespace ispc;

#include "../../common/timing.h"

#define NSUBSAMPLES 2

extern void ao_serial(int w, int h, int nsubsamples, float image[]);

static unsigned int test_iterations[] = {3, 7, 1};
static unsigned int width, height;
static unsigned char *img;
static float *fimg;

static unsigned char clamp(float f) {
    int i = (int)(f * 255.5);

    if (i < 0)
        i = 0;
    if (i > 255)
        i = 255;

    return (unsigned char)i;
}

static void savePPM(const char *fname, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img[3 * (y * w + x) + 0] = clamp(fimg[3 * (y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 * (y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 * (y * w + x) + 2]);
        }
    }

    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror(fname);
        exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
    printf("Wrote image file %s\n", fname);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("%s\n", argv[0]);
        printf("Usage: ao [width] [height] [ispc iterations] [tasks iterations] [serial iterations]\n");
        exit(-1);
    } else {
        if (argc == 6) {
            for (int i = 0; i < 3; i++) {
                test_iterations[i] = atoi(argv[3 + i]);
            }
        }
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    // Allocate space for output images
    img = new unsigned char[width * height * 3];
    fimg = new float[width * height * 3];

    //
    // Run the ispc path, test_iterations times, and report the minimum
    // time for any of them.
    //
    double minTimeISPC = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        assert(NSUBSAMPLES == 2);

        reset_and_start_timer();
        ao_ispc(width, height, NSUBSAMPLES, fimg);
        double t = get_elapsed_mcycles();
        printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", t);
        minTimeISPC = std::min(minTimeISPC, t);
    }

    // Report results and save image
    printf("[aobench ispc]:\t\t\t[%.3f] million cycles (%d x %d image)\n", minTimeISPC, width, height);
    savePPM("ao-ispc.ppm", width, height);

    //
    // Run the ispc + tasks path, test_iterations times, and report the
    // minimum time for any of them.
    //
    double minTimeISPCTasks = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        assert(NSUBSAMPLES == 2);

        reset_and_start_timer();
        ao_ispc_tasks(width, height, NSUBSAMPLES, fimg);
        double t = get_elapsed_mcycles();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", t);
        minTimeISPCTasks = std::min(minTimeISPCTasks, t);
    }

    // Report results and save image
    printf("[aobench ispc + tasks]:\t\t[%.3f] million cycles (%d x %d image)\n", minTimeISPCTasks, width, height);
    savePPM("ao-ispc-tasks.ppm", width, height);

    //
    // Run the serial path, again test_iteration times, and report the
    // minimum time.
    //
    double minTimeSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[2]; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        reset_and_start_timer();
        ao_serial(width, height, NSUBSAMPLES, fimg);
        double t = get_elapsed_mcycles();
        printf("@time of serial run:\t\t\t\t[%.3f] million cycles\n", t);
        minTimeSerial = std::min(minTimeSerial, t);
    }

    // Report more results, save another image...
    printf("[aobench serial]:\t\t[%.3f] million cycles (%d x %d image)\n", minTimeSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", minTimeSerial / minTimeISPC,
           minTimeSerial / minTimeISPCTasks);
    savePPM("ao-serial.ppm", width, height);

    return 0;
}
