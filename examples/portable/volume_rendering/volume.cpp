/*
  Copyright (c) 2011-2014, Intel Corporation
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
#include <algorithm>
#include "timing.h"
#include "ispc_malloc.h"
#include "volume_ispc.h"
using namespace ispc;

/* Write a PPM image file with the image */
static void
writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0.f) v = 0.f;
        else if (v > 255.f) v = 255.f;
        unsigned char c = (unsigned char)v;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}


/* Load image and viewing parameters from a camera data file.
   FIXME: we should add support to be able to specify viewing parameters
   in the program here directly. */
static void
loadCamera(const char *fn, int *width, int *height, float raster2camera[4][4],
           float camera2world[4][4]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }
    if (fscanf(f, "%d %d", width, height) != 2) {
        fprintf(stderr, "Unexpected end of file in camera file\n");
        exit(1);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &raster2camera[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fscanf(f, "%f", &camera2world[i][j]) != 1) {
                fprintf(stderr, "Unexpected end of file in camera file\n");
                exit(1);
            }
        }
    }
    fclose(f);
}


/* Load a volume density file.  Expects the number of x, y, and z samples
   as the first three values (as integer strings), then x*y*z
   floating-point values (also as strings) to give the densities.  */
static float *
loadVolume(const char *fn, int n[3]) {
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }

    if (fscanf(f, "%d %d %d", &n[0], &n[1], &n[2]) != 3) {
        fprintf(stderr, "Couldn't find resolution at start of density file\n");
        exit(1);
    }

    int count = n[0] * n[1] * n[2];
    float *v = new float[count];
    for (int i = 0; i < count; ++i) {
        if (fscanf(f, "%f", &v[i]) != 1) {
            fprintf(stderr, "Unexpected end of file at %d'th density value\n", i);
            exit(1);
        }
    }

    return v;
}


int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 7, 1};
    if (argc < 3) {
        fprintf(stderr, "usage: volume <camera.dat> <volume_density.vol> [ispc iterations] [tasks iterations] [serial iterations]\n");
        return 1;
    }
    if (argc == 6) {
        for (int i = 0; i < 3; i++) {
            test_iterations[i] = atoi(argv[3 + i]);
        }
    }

    //
    // Load viewing data and the volume density data
    //
    int width, height;

    float *camera2world_ispc = new float[4*4];
    float *raster2camera_ispc = new float[4*4];
    float (*camera2world )[4] = (float (*)[4])camera2world_ispc;
    float (*raster2camera)[4] = (float (*)[4])raster2camera_ispc;

    loadCamera(argv[1], &width, &height, raster2camera, camera2world);
    float *image = new float[width*height];

    int *n = new int[3];
    float *density = loadVolume(argv[2], n);

    // Clear out the buffer
    for (int i = 0; i < width * height; ++i)
        image[i] = 0.;

    //
    // Compute the image using the ispc implementation that also uses
    // tasks; report the minimum time of three runs.
    //
    double minISPCtasks = 1e30;
    for (int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        volume_ispc_tasks(density, n, raster2camera, camera2world,
                          width, height, image);
        double dt = get_elapsed_msec();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] msec\n", dt);
        minISPCtasks = std::min(minISPCtasks, dt);
    }

    printf("[volume ispc + tasks]:\t\t[%.3f] msec\n", minISPCtasks);
    writePPM(image, width, height, "volume-ispc-tasks.ppm");

    return 0;
}
