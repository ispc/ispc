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
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include <math.h>
#include <map>
#include <string>
#include <algorithm>
#include <sys/types.h>

#include "ao_ispc.h"
using namespace ispc;

#include "instrument.h"
#include "../timing.h"

#define NSUBSAMPLES        2

static unsigned int test_iterations;
static unsigned int width, height;
static unsigned char *img;
static float *fimg;


static unsigned char
clamp(float f)
{
    int i = (int)(f * 255.5);

    if (i < 0) i = 0;
    if (i > 255) i = 255;

    return (unsigned char)i;
}


static void
savePPM(const char *fname, int w, int h)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)  {
            img[3 * (y * w + x) + 0] = clamp(fimg[3 *(y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 *(y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 *(y * w + x) + 2]);
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
}


// Allocate memory with 64-byte alignment.
float *
AllocAligned(int size) {
#if defined(_WIN32) || defined(_WIN64)
    return (float *)_aligned_malloc(size, 64);
#elif defined (__APPLE__)
    // Allocate excess memory to ensure an aligned pointer can be returned
    void *mem = malloc(size + (64-1) + sizeof(void*));
    char *amem = ((char*)mem) + sizeof(void*);
    amem += 64 - (reinterpret_cast<uint64_t>(amem) & (64 - 1));
    ((void**)amem)[-1] = mem;
    return (float *)amem;
#else
    return (float *)memalign(64, size);
#endif
}


int main(int argc, char **argv)
{
    if (argc != 4) {
        printf ("%s\n", argv[0]);
        printf ("Usage: ao [num test iterations] [width] [height]\n");
        getchar();
        exit(-1);
    }
    else {
        test_iterations = atoi(argv[1]);
        width = atoi (argv[2]);
        height = atoi (argv[3]);
    }

    // Allocate space for output images
    img = (unsigned char *)AllocAligned(width * height * 3);
    fimg = (float *)AllocAligned(sizeof(float) * width * height * 3);

    ao_ispc(width, height, NSUBSAMPLES, fimg);

    savePPM("ao-ispc.ppm", width, height); 

    ISPCPrintInstrument();

    return 0;
}
