/*
  Copyright (c) 2011-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef DEFERRED_H
#define DEFERRED_H

// Currently tile widths must be a multiple of SIMD width (i.e. 8 for ispc sse4x2)!
#define MIN_TILE_WIDTH 16
#define MIN_TILE_HEIGHT 16
#define MAX_LIGHTS 1024

enum InputDataArraysEnum {
    idaZBuffer = 0,
    idaNormalEncoded_x,
    idaNormalEncoded_y,
    idaSpecularAmount,
    idaSpecularPower,
    idaAlbedo_x,
    idaAlbedo_y,
    idaAlbedo_z,
    idaLightPositionView_x,
    idaLightPositionView_y,
    idaLightPositionView_z,
    idaLightAttenuationBegin,
    idaLightColor_x,
    idaLightColor_y,
    idaLightColor_z,
    idaLightAttenuationEnd,

    idaNum
};

#ifndef ISPC

#include "kernels_ispc.h"
#include <stdint.h>

#define ALIGNMENT_BYTES 64

#define MAX_LIGHTS 1024

#define VISUALIZE_LIGHT_COUNT 0

struct InputData {
    ispc::InputHeader header;
    ispc::InputDataArrays arrays;
    uint8_t *chunk;
};

struct Framebuffer {
    Framebuffer(int width, int height);
    ~Framebuffer();

    void clear();

    uint8_t *r, *g, *b;

  private:
    int nPixels;
    Framebuffer(const Framebuffer &);
    Framebuffer &operator=(const Framebuffer *);
};

InputData *CreateInputDataFromFile(const char *path);
void DeleteInputData(InputData *input);
void WriteFrame(const char *filename, const InputData *input, const Framebuffer &framebuffer);
void InitDynamicC(InputData *input);
void DispatchDynamicC(InputData *input, Framebuffer *framebuffer);

#endif // !ISPC

#endif // DEFERRED_H
