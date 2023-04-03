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
#include "rt_ispc.h"
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

using namespace ispc;

typedef unsigned int uint;

extern void raytrace_serial(int width, int height, int baseWidth, int baseHeight, const float raster2camera[4][4],
                            const float camera2world[4][4], float image[], int id[], const LinearBVHNode nodes[],
                            const Triangle triangles[]);

static void writeImage(int *idImage, float *depthImage, int width, int height, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // use the bits from the object id of the hit object to make a
            // random color
            int id = idImage[y * width + x];
            unsigned char r = 0, g = 0, b = 0;

            for (int i = 0; i < 8; ++i) {
                // extract bit 3*i for red, 3*i+1 for green, 3*i+2 for blue
                int rbit = (id & (1 << (3 * i))) >> (3 * i);
                int gbit = (id & (1 << (3 * i + 1))) >> (3 * i + 1);
                int bbit = (id & (1 << (3 * i + 2))) >> (3 * i + 2);
                // and then set the bits of the colors starting from the
                // high bits...
                r |= rbit << (7 - i);
                g |= gbit << (7 - i);
                b |= bbit << (7 - i);
            }
            fputc(r, f);
            fputc(g, f);
            fputc(b, f);
        }
    }
    fclose(f);
    printf("Wrote image file %s\n", filename);
}

static void usage() {
    fprintf(stderr,
            "rt <scene name base> [--scale=<factor>] [ispc iterations] [tasks iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 7, 1};
    float scale = 1.f;
    const char *filename = nullptr;
    if (argc < 2)
        usage();
    filename = argv[1];
    if (argc > 2) {
        if (strncmp(argv[2], "--scale=", 8) == 0) {
            scale = atof(argv[2] + 8);
        }
    }
    if ((argc == 6) || (argc == 5)) {
        for (int i = 0; i < 3; i++) {
            test_iterations[i] = atoi(argv[argc - 3 + i]);
        }
    }

#define READ(var, n)                                                                                                   \
    if (fread(&(var), sizeof(var), n, f) != (unsigned int)n) {                                                         \
        fprintf(stderr, "Unexpected EOF reading scene file\n");                                                        \
        return 1;                                                                                                      \
    } else /* eat ; */

    //
    // Read the camera specification information from the camera file
    //
    constexpr std::size_t FILENAME_MAX_LEN{1000UL};
    char filename_buf[FILENAME_MAX_LEN];
    snprintf(filename_buf, FILENAME_MAX_LEN, "%s.camera", filename);
    FILE *f = fopen(filename_buf, "rb");
    if (!f) {
        perror(filename_buf);
        return 1;
    }

    //
    // Nothing fancy, and trouble if we run on a big-endian system, just
    // fread in the bits
    //
    int baseWidth, baseHeight;
    float camera2world[4][4], raster2camera[4][4];
    READ(baseWidth, 1);
    READ(baseHeight, 1);
    READ(camera2world[0][0], 16);
    READ(raster2camera[0][0], 16);

    //
    // Read in the serialized BVH
    //
    snprintf(filename_buf, FILENAME_MAX_LEN, "%s.bvh", filename);
    f = fopen(filename_buf, "rb");
    if (!f) {
        perror(filename_buf);
        return 1;
    }

    // The BVH file starts with an int that gives the total number of BVH
    // nodes
    uint nNodes;
    READ(nNodes, 1);

    LinearBVHNode *nodes = new LinearBVHNode[nNodes];
    for (unsigned int i = 0; i < nNodes; ++i) {
        // Each node is 6x floats for a boox, then an integer for an offset
        // to the second child node, then an integer that encodes the type
        // of node, the total number of int it if a leaf node, etc.
        float b[6];
        READ(b[0], 6);
        nodes[i].bounds[0][0] = b[0];
        nodes[i].bounds[0][1] = b[1];
        nodes[i].bounds[0][2] = b[2];
        nodes[i].bounds[1][0] = b[3];
        nodes[i].bounds[1][1] = b[4];
        nodes[i].bounds[1][2] = b[5];
        READ(nodes[i].offset, 1);
        READ(nodes[i].nPrimitives, 1);
        READ(nodes[i].splitAxis, 1);
        READ(nodes[i].pad, 1);
    }

    // And then read the triangles
    uint nTris;
    READ(nTris, 1);
    Triangle *triangles = new Triangle[nTris];
    for (uint i = 0; i < nTris; ++i) {
        // 9x floats for the 3 vertices
        float v[9];
        READ(v[0], 9);
        float *vp = v;
        for (int j = 0; j < 3; ++j) {
            triangles[i].p[j][0] = *vp++;
            triangles[i].p[j][1] = *vp++;
            triangles[i].p[j][2] = *vp++;
        }
        // And create an object id
        triangles[i].id = i + 1;
    }
    fclose(f);

    int height = int(baseHeight * scale);
    int width = int(baseWidth * scale);

    // allocate images; one to hold hit object ids, one to hold depth to
    // the first interseciton
    int *id = new int[width * height];
    float *image = new float[width * height];

    //
    // Run 3 iterations with ispc + 1 core, record the minimum time
    //
    double minTimeISPC = 1e30;
    for (uint i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        raytrace_ispc(width, height, baseWidth, baseHeight, raster2camera, camera2world, image, id, nodes, triangles);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", dt);
        minTimeISPC = std::min(dt, minTimeISPC);
    }
    printf("[rt ispc, 1 core]:\t\t[%.3f] million cycles for %d x %d image\n", minTimeISPC, width, height);

    writeImage(id, image, width, height, "rt-ispc-1core.ppm");

    memset(id, 0, width * height * sizeof(int));
    memset(image, 0, width * height * sizeof(float));

    //
    // Run 3 iterations with ispc + 1 core, record the minimum time
    //
    double minTimeISPCtasks = 1e30;
    for (uint i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        raytrace_ispc_tasks(width, height, baseWidth, baseHeight, raster2camera, camera2world, image, id, nodes,
                            triangles);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", dt);
        minTimeISPCtasks = std::min(dt, minTimeISPCtasks);
    }
    printf("[rt ispc + tasks]:\t\t[%.3f] million cycles for %d x %d image\n", minTimeISPCtasks, width, height);

    writeImage(id, image, width, height, "rt-ispc-tasks.ppm");

    memset(id, 0, width * height * sizeof(int));
    memset(image, 0, width * height * sizeof(float));

    //
    // And 3 iterations with the serial implementation, reporting the
    // minimum time.
    //
    double minTimeSerial = 1e30;
    for (uint i = 0; i < test_iterations[2]; ++i) {
        reset_and_start_timer();
        raytrace_serial(width, height, baseWidth, baseHeight, raster2camera, camera2world, image, id, nodes, triangles);
        double dt = get_elapsed_mcycles();
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
        minTimeSerial = std::min(dt, minTimeSerial);
    }
    printf("[rt serial]:\t\t\t[%.3f] million cycles for %d x %d image\n", minTimeSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", minTimeSerial / minTimeISPC,
           minTimeSerial / minTimeISPCtasks);

    writeImage(id, image, width, height, "rt-serial.ppm");

    return 0;
}
