/*
  Copyright (c) 2011, Intel Corporation
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
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <stdint.h>
#include <algorithm>
#include <assert.h>
#include <vector>
#ifdef ISPC_IS_WINDOWS
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif
#ifdef ISPC_IS_LINUX
  #include <malloc.h>
#endif
#include "deferred.h"
#include "../timing.h"

///////////////////////////////////////////////////////////////////////////

static void *
lAlignedMalloc(size_t size, int32_t alignment) {
#ifdef ISPC_IS_WINDOWS
    return _aligned_malloc(size, alignment);
#endif
#ifdef ISPC_IS_LINUX
    return memalign(alignment, size);
#endif
#ifdef ISPC_IS_APPLE
    void *mem = malloc(size + (alignment-1) + sizeof(void*));
    char *amem = ((char*)mem) + sizeof(void*);
    amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) &
                                        (alignment - 1)));
    ((void**)amem)[-1] = mem;
    return amem;
#endif
}


static void
lAlignedFree(void *ptr) {
#ifdef ISPC_IS_WINDOWS
    _aligned_free(ptr);
#endif
#ifdef ISPC_IS_LINUX
    free(ptr);
#endif
#ifdef ISPC_IS_APPLE
    free(((void**)ptr)[-1]);
#endif
}


Framebuffer::Framebuffer(int width, int height) {
    nPixels = width*height;
    r = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
    g = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
    b = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
}


Framebuffer::~Framebuffer() {
    lAlignedFree(r);
    lAlignedFree(g);
    lAlignedFree(b);
}


void
Framebuffer::clear() {
    memset(r, 0, nPixels);
    memset(g, 0, nPixels);
    memset(b, 0, nPixels);
}


InputData *
CreateInputDataFromFile(const char *path) {
    FILE *in = fopen(path, "rb");
    if (!in) return 0;

    InputData *input = new InputData;

    // Load header
    if (fread(&input->header, sizeof(ispc::InputHeader), 1, in) != 1) {
        fprintf(stderr, "Preumature EOF reading file \"%s\"\n", path);
        return NULL;
    }

    // Load data chunk and update pointers
    input->chunk = (uint8_t *)lAlignedMalloc(input->header.inputDataChunkSize, 
                                             ALIGNMENT_BYTES);
    if (fread(input->chunk, input->header.inputDataChunkSize, 1, in) != 1) {
        fprintf(stderr, "Preumature EOF reading file \"%s\"\n", path);
        return NULL;
    }
    
    input->arrays.zBuffer =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaZBuffer]];
    input->arrays.normalEncoded_x =
        (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaNormalEncoded_x]];
    input->arrays.normalEncoded_y =
        (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaNormalEncoded_y]];
    input->arrays.specularAmount =
        (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaSpecularAmount]];
    input->arrays.specularPower =
        (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaSpecularPower]];
    input->arrays.albedo_x =
        (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_x]];
    input->arrays.albedo_y =
        (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_y]];
    input->arrays.albedo_z =
        (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_z]];
    input->arrays.lightPositionView_x =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_x]];
    input->arrays.lightPositionView_y =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_y]];
    input->arrays.lightPositionView_z =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_z]];
    input->arrays.lightAttenuationBegin =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightAttenuationBegin]];
    input->arrays.lightColor_x =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_x]];
    input->arrays.lightColor_y =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_y]];
    input->arrays.lightColor_z =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_z]];
    input->arrays.lightAttenuationEnd =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightAttenuationEnd]];

    fclose(in);
    return input;
}


void DeleteInputData(InputData *input) {
    lAlignedFree(input->chunk);
}


void WriteFrame(const char *filename, const InputData *input,
                const Framebuffer &framebuffer) {
    // Deswizzle and copy to RGBA output
    // Doesn't need to be fast... only happens once
    size_t imageBytes = 3 * input->header.framebufferWidth * 
        input->header.framebufferHeight;
    uint8_t* framebufferAOS = (uint8_t *)lAlignedMalloc(imageBytes, ALIGNMENT_BYTES);
    memset(framebufferAOS, 0, imageBytes);

    for (int i = 0; i < input->header.framebufferWidth * 
                        input->header.framebufferHeight; ++i) {
        framebufferAOS[3 * i + 0] = framebuffer.r[i];
        framebufferAOS[3 * i + 1] = framebuffer.g[i];
        framebufferAOS[3 * i + 2] = framebuffer.b[i];
    }
    
    // Write out simple PPM file
    FILE *out = fopen(filename, "wb");
    fprintf(out, "P6 %d %d 255\n", input->header.framebufferWidth, 
            input->header.framebufferHeight);
    fwrite(framebufferAOS, imageBytes, 1, out);
    fclose(out);

    lAlignedFree(framebufferAOS);
}
