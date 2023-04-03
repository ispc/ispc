/*
  Copyright (c) 2011-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define ISPC_IS_WINDOWS
#elif defined(__linux__) || defined(__FreeBSD__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#else
#error "Host OS was not detected"
#endif

#include <algorithm>
#include <assert.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>
#ifdef ISPC_IS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif
#include "../../common/timing.h"
#include "deferred.h"

///////////////////////////////////////////////////////////////////////////

static void *lAlignedMalloc(size_t size, int32_t alignment) {
#ifdef ISPC_IS_WINDOWS
    return _aligned_malloc(size, alignment);
#elif defined ISPC_IS_LINUX
    return memalign(alignment, size);
#elif defined ISPC_IS_APPLE
    void *mem = malloc(size + (alignment - 1) + sizeof(void *));
    char *amem = ((char *)mem) + sizeof(void *);
    amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) & (alignment - 1)));
    ((void **)amem)[-1] = mem;
    return amem;
#else
#error "Host OS was not detected"
#endif
}

static void lAlignedFree(void *ptr) {
#ifdef ISPC_IS_WINDOWS
    _aligned_free(ptr);
#elif defined ISPC_IS_LINUX
    free(ptr);
#elif defined ISPC_IS_APPLE
    free(((void **)ptr)[-1]);
#else
#error "Host OS was not detected"
#endif
}

Framebuffer::Framebuffer(int width, int height) {
    nPixels = width * height;
    r = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
    g = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
    b = (uint8_t *)lAlignedMalloc(nPixels, ALIGNMENT_BYTES);
}

Framebuffer::~Framebuffer() {
    lAlignedFree(r);
    lAlignedFree(g);
    lAlignedFree(b);
}

void Framebuffer::clear() {
    memset(r, 0, nPixels);
    memset(g, 0, nPixels);
    memset(b, 0, nPixels);
}

InputData *CreateInputDataFromFile(const char *path) {
    FILE *in = fopen(path, "rb");
    if (!in)
        return 0;

    InputData *input = new InputData;

    // Load header
    if (fread(&input->header, sizeof(ispc::InputHeader), 1, in) != 1) {
        fprintf(stderr, "Preumature EOF reading file \"%s\"\n", path);
        fclose(in);
        delete input;
        return nullptr;
    }

    // Load data chunk and update pointers
    input->chunk = (uint8_t *)lAlignedMalloc(input->header.inputDataChunkSize, ALIGNMENT_BYTES);
    if (fread(input->chunk, input->header.inputDataChunkSize, 1, in) != 1) {
        fprintf(stderr, "Preumature EOF reading file \"%s\"\n", path);
        fclose(in);
        delete input;
        return nullptr;
    }

    input->arrays.zBuffer = (float *)&input->chunk[input->header.inputDataArrayOffsets[idaZBuffer]];
    input->arrays.normalEncoded_x = (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaNormalEncoded_x]];
    input->arrays.normalEncoded_y = (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaNormalEncoded_y]];
    input->arrays.specularAmount = (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaSpecularAmount]];
    input->arrays.specularPower = (uint16_t *)&input->chunk[input->header.inputDataArrayOffsets[idaSpecularPower]];
    input->arrays.albedo_x = (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_x]];
    input->arrays.albedo_y = (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_y]];
    input->arrays.albedo_z = (uint8_t *)&input->chunk[input->header.inputDataArrayOffsets[idaAlbedo_z]];
    input->arrays.lightPositionView_x =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_x]];
    input->arrays.lightPositionView_y =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_y]];
    input->arrays.lightPositionView_z =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightPositionView_z]];
    input->arrays.lightAttenuationBegin =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightAttenuationBegin]];
    input->arrays.lightColor_x = (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_x]];
    input->arrays.lightColor_y = (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_y]];
    input->arrays.lightColor_z = (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightColor_z]];
    input->arrays.lightAttenuationEnd =
        (float *)&input->chunk[input->header.inputDataArrayOffsets[idaLightAttenuationEnd]];

    fclose(in);
    return input;
}

void DeleteInputData(InputData *input) { lAlignedFree(input->chunk); }

void WriteFrame(const char *filename, const InputData *input, const Framebuffer &framebuffer) {
    // Deswizzle and copy to RGBA output
    // Doesn't need to be fast... only happens once
    size_t imageBytes = 3 * input->header.framebufferWidth * input->header.framebufferHeight;
    uint8_t *framebufferAOS = (uint8_t *)lAlignedMalloc(imageBytes, ALIGNMENT_BYTES);
    memset(framebufferAOS, 0, imageBytes);

    for (int i = 0; i < input->header.framebufferWidth * input->header.framebufferHeight; ++i) {
        framebufferAOS[3 * i + 0] = framebuffer.r[i];
        framebufferAOS[3 * i + 1] = framebuffer.g[i];
        framebufferAOS[3 * i + 2] = framebuffer.b[i];
    }

    // Write out simple PPM file
    FILE *out = fopen(filename, "wb");
    if (!out) {
        printf("Couldn't open a file '%s'\n", filename);
        exit(1);
    }
    fprintf(out, "P6 %d %d 255\n", input->header.framebufferWidth, input->header.framebufferHeight);
    fwrite(framebufferAOS, imageBytes, 1, out);
    fclose(out);

    lAlignedFree(framebufferAOS);
}
