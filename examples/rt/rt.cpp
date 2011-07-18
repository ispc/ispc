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
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <sys/types.h>
#include "../timing.h"
#include "../cpuid.h"
#include "rt_ispc.h"

using namespace ispc;

typedef unsigned int uint;

extern void raytrace_serial(int width, int height, const float raster2camera[4][4], 
                            const float camera2world[4][4], float image[],
                            int id[], const LinearBVHNode nodes[],
                            const Triangle triangles[]);


static void writeImage(int *idImage, float *depthImage, int width, int height,
                       const char *filename) {
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
                int rbit = (id & (1 << (3*i)))   >> (3*i);
                int gbit = (id & (1 << (3*i+1))) >> (3*i+1);
                int bbit = (id & (1 << (3*i+2))) >> (3*i+2);
                // and then set the bits of the colors starting from the
                // high bits...
                r |= rbit << (7-i);
                g |= gbit << (7-i);
                b |= bbit << (7-i);
            }
            fputc(r, f);
            fputc(g, f);
            fputc(b, f);
        }
    }            
    fclose(f);
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


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: rt <filename base>\n");
        exit(1);
    }

    ensureTargetISAIsSupported();

#define READ(var, n)                                            \
    if (fread(&(var), sizeof(var), n, f) != (unsigned int)n) {  \
        fprintf(stderr, "Unexpected EOF reading scene file\n"); \
        return 1;                                               \
    } else /* eat ; */                                                     

    //
    // Read the camera specification information from the camera file
    //
    char fnbuf[1024];
    sprintf(fnbuf, "%s.camera", argv[1]);
    FILE *f = fopen(fnbuf, "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }

    //
    // Nothing fancy, and trouble if we run on a big-endian system, just
    // fread in the bits
    //
    int width, height;
    float camera2world[4][4], raster2camera[4][4];
    READ(width, 1);
    READ(height, 1);
    READ(camera2world[0][0], 16);
    READ(raster2camera[0][0], 16);

    //
    // Read in the serialized BVH 
    //
    sprintf(fnbuf, "%s.bvh", argv[1]);
    f = fopen(fnbuf, "rb");
    if (!f) {
        perror(argv[2]);
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
        nodes[i].bounds[0].v[0] = b[0];
        nodes[i].bounds[0].v[1] = b[1];
        nodes[i].bounds[0].v[2] = b[2];
        nodes[i].bounds[1].v[0] = b[3];
        nodes[i].bounds[1].v[1] = b[4];
        nodes[i].bounds[1].v[2] = b[5];
        READ(nodes[i].offset, 1);
        READ(nodes[i].primsAxis, 1);
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
            triangles[i].p[j].v[0] = *vp++;
            triangles[i].p[j].v[1] = *vp++;
            triangles[i].p[j].v[2] = *vp++;
        }
        // And create an object id
        triangles[i].id = i+1;
    }
    fclose(f);

    // round image resolution up to multiple of 4 to makethings easy for
    // the code that assigns pixels to ispc program instances
    height = (height + 3) & ~3;
    width = (width + 3) & ~3;

    // allocate images; one to hold hit object ids, one to hold depth to
    // the first interseciton
    int *id = new int[width*height];
    float *image = new float[width*height];

    //
    // Run 3 iterations with ispc, record the minimum time
    //
    double minTimeISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        raytrace(width, height, raster2camera, camera2world, 
                 image, id, nodes, triangles);
        double dt = get_elapsed_mcycles();
        minTimeISPC = std::min(dt, minTimeISPC);
    }
    printf("[rt ispc]:\t\t\t[%.3f] million cycles for %d x %d image\n", minTimeISPC, width, height);

    writeImage(id, image, width, height, "rt-ispc.ppm");

    //
    // And 3 iterations with the serial implementation, reporting the
    // minimum time.
    //
    double minTimeSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        reset_and_start_timer();
        raytrace_serial(width, height, raster2camera, camera2world, 
                        image, id, nodes, triangles);
        double dt = get_elapsed_mcycles();
        minTimeSerial = std::min(dt, minTimeSerial);
    }
    printf("[rt serial]:\t\t\t[%.3f] million cycles for %d x %d image\n", 
           minTimeSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minTimeSerial / minTimeISPC);

    writeImage(id, image, width, height, "rt-serial.ppm");

    return 0;
}
