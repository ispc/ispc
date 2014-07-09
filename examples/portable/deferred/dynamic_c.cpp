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

#include "deferred.h"
#include "kernels_ispc.h"
#include <algorithm>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#ifdef _MSC_VER
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif // ISPC_IS_LINUX

// Currently tile widths must be a multiple of SIMD width (i.e. 8 for ispc sse4x2)!
#ifndef MIN_TILE_WIDTH
#define MIN_TILE_WIDTH 16
#endif
#ifndef MIN_TILE_HEIGHT
#define MIN_TILE_HEIGHT 16
#endif


#define DYNAMIC_TREE_LEVELS 5
// If this is set to 1 then the result will be identical to the static version
#define DYNAMIC_MIN_LIGHTS_TO_SUBDIVIDE 1

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


static void
ComputeZBounds(int tileStartX, int tileEndX,
               int tileStartY, int tileEndY,
               // G-buffer data
               float zBuffer[],
               int gBufferWidth,
               // Camera data
               float cameraProj_33, float cameraProj_43,
               float cameraNear, float cameraFar,
               // Output
               float *minZ, float *maxZ)
{
    // Find Z bounds
    float laneMinZ = cameraFar;
    float laneMaxZ = cameraNear;
    for (int y = tileStartY; y < tileEndY; ++y) {
        for (int x = tileStartX; x < tileEndX; ++x) {
            // Unproject depth buffer Z value into view space
            float z = zBuffer[(y * gBufferWidth + x)];
            float viewSpaceZ = cameraProj_43 / (z - cameraProj_33);

            // Work out Z bounds for our samples
            // Avoid considering skybox/background or otherwise invalid pixels
            if ((viewSpaceZ < cameraFar) && (viewSpaceZ >= cameraNear)) {
                laneMinZ = std::min(laneMinZ, viewSpaceZ);
                laneMaxZ = std::max(laneMaxZ, viewSpaceZ);
            }
        }
    }
    *minZ = laneMinZ;
    *maxZ = laneMaxZ;
}


static void
ComputeZBoundsRow(int tileY, int tileWidth, int tileHeight,
                  int numTilesX, int numTilesY,
                  // G-buffer data
                  float zBuffer[],
                  int gBufferWidth,
                  // Camera data
                  float cameraProj_33, float cameraProj_43,
                  float cameraNear, float cameraFar,
                  // Output
                  float minZArray[],
                  float maxZArray[])
{
    for (int tileX = 0; tileX < numTilesX; ++tileX) {
        float minZ, maxZ;
        ComputeZBounds(tileX * tileWidth, tileX * tileWidth + tileWidth,
                       tileY * tileHeight, tileY * tileHeight + tileHeight,
                       zBuffer, gBufferWidth, cameraProj_33, cameraProj_43,
                       cameraNear, cameraFar, &minZ, &maxZ);
        minZArray[tileX] = minZ;
        maxZArray[tileX] = maxZ;
    }
}


class MinMaxZTree
{
public:
    // Currently (min) tile dimensions must divide gBuffer dimensions evenly
    // Levels must be small enough that neither dimension goes below one tile
    MinMaxZTree(
        int tileWidth, int tileHeight, int levels,
        int gBufferWidth, int gBufferHeight)
        : mTileWidth(tileWidth), mTileHeight(tileHeight), mLevels(levels)
    {
        mNumTilesX = gBufferWidth / mTileWidth;
        mNumTilesY = gBufferHeight / mTileHeight;

        // Allocate arrays
        mMinZArrays = (float **)lAlignedMalloc(sizeof(float *) * mLevels, 16);
        mMaxZArrays = (float **)lAlignedMalloc(sizeof(float *) * mLevels, 16);
        for (int i = 0; i < mLevels; ++i) {
            int x = NumTilesX(i);
            int y = NumTilesY(i);
            assert(x > 0);
            assert(y > 0);
            // NOTE: If the following two asserts fire it probably means that
            // the base tile dimensions do not evenly divide the G-buffer dimensions
            assert(x * (mTileWidth << i) >= gBufferWidth);
            assert(y * (mTileHeight << i) >= gBufferHeight);
            mMinZArrays[i] = (float *)lAlignedMalloc(sizeof(float) * x * y, 16);
            mMaxZArrays[i] = (float *)lAlignedMalloc(sizeof(float) * x * y, 16);
        }
    }

    void Update(float *zBuffer, int gBufferPitchInElements,
        float cameraProj_33, float cameraProj_43,
        float cameraNear, float cameraFar)
    {
        for (int tileY = 0; tileY < mNumTilesY; ++tileY) {
            ComputeZBoundsRow(tileY, mTileWidth, mTileHeight, mNumTilesX, mNumTilesY,
                              zBuffer, gBufferPitchInElements,
                              cameraProj_33, cameraProj_43, cameraNear, cameraFar,
                              mMinZArrays[0] + (tileY * mNumTilesX),
                              mMaxZArrays[0] + (tileY * mNumTilesX));
        }

        // Generate other levels
        for (int level = 1; level < mLevels; ++level) {
            int destTilesX = NumTilesX(level);
            int destTilesY = NumTilesY(level);
            int srcLevel = level - 1;
            int srcTilesX = NumTilesX(srcLevel);
            int srcTilesY = NumTilesY(srcLevel);
            for (int y = 0; y < destTilesY; ++y) {
                for (int x = 0; x < destTilesX; ++x) {
                    int srcX = x << 1;
                    int srcY = y << 1;
                    // NOTE: Ugly branches to deal with non-multiple dimensions at some levels
                    // TODO: SSE branchless min/max is probably better...
                    float minZ = mMinZArrays[srcLevel][(srcY) * srcTilesX + (srcX)];
                    float maxZ = mMaxZArrays[srcLevel][(srcY) * srcTilesX + (srcX)];
                    if (srcX + 1 < srcTilesX) {
                        minZ = std::min(minZ, mMinZArrays[srcLevel][(srcY) * srcTilesX +
                                                                    (srcX + 1)]);
                        maxZ = std::max(maxZ, mMaxZArrays[srcLevel][(srcY) * srcTilesX +
                                                                    (srcX + 1)]);
                        if (srcY + 1 < srcTilesY) {
                            minZ = std::min(minZ, mMinZArrays[srcLevel][(srcY + 1) * srcTilesX +
                                                                        (srcX + 1)]);
                            maxZ = std::max(maxZ, mMaxZArrays[srcLevel][(srcY + 1) * srcTilesX +
                                                                        (srcX + 1)]);
                        }
                    }
                    if (srcY + 1 < srcTilesY) {
                        minZ = std::min(minZ, mMinZArrays[srcLevel][(srcY + 1) * srcTilesX +
                                                                    (srcX    )]);
                        maxZ = std::max(maxZ, mMaxZArrays[srcLevel][(srcY + 1) * srcTilesX +
                                                                    (srcX    )]);
                    }
                    mMinZArrays[level][y * destTilesX + x] = minZ;
                    mMaxZArrays[level][y * destTilesX + x] = maxZ;
                }
            }
        }
    }

    ~MinMaxZTree() {
        for (int i = 0; i < mLevels; ++i) {
            lAlignedFree(mMinZArrays[i]);
            lAlignedFree(mMaxZArrays[i]);
        }
        lAlignedFree(mMinZArrays);
        lAlignedFree(mMaxZArrays);
    }

    int Levels() const { return mLevels; }

    // These round UP, so beware that the last tile for a given level may not be completely full
    // TODO: Verify this...
    int NumTilesX(int level = 0) const { return (mNumTilesX + (1 << level) - 1) >> level; }
    int NumTilesY(int level = 0) const { return (mNumTilesY + (1 << level) - 1) >> level; }
    int TileWidth(int level = 0) const { return (mTileWidth << level); }
    int TileHeight(int level = 0) const { return (mTileHeight << level); }

    float MinZ(int level, int tileX, int tileY) const {
        return mMinZArrays[level][tileY * NumTilesX(level) + tileX];
    }
    float MaxZ(int level, int tileX, int tileY) const {
        return mMaxZArrays[level][tileY * NumTilesX(level) + tileX];
    }

private:
    int mTileWidth;
    int mTileHeight;
    int mLevels;
    int mNumTilesX;
    int mNumTilesY;

    // One array for each "level" in the tree
    float **mMinZArrays;
    float **mMaxZArrays;
};

static MinMaxZTree *gMinMaxZTree = 0;

void InitDynamicC(InputData *input) {
    gMinMaxZTree =
        new MinMaxZTree(MIN_TILE_WIDTH, MIN_TILE_HEIGHT, DYNAMIC_TREE_LEVELS,
                        input->header.framebufferWidth,
                        input->header.framebufferHeight);
}


/* We're going to split a tile into 4 sub-tiles.  This function
   reclassifies the tile's lights with respect to the sub-tiles. */
static void
SplitTileMinMax(
    int tileMidX, int tileMidY,
    // Subtile data (00, 10, 01, 11)
    float subtileMinZ[],
    float subtileMaxZ[],
    // G-buffer data
    int gBufferWidth, int gBufferHeight,
    // Camera data
    float cameraProj_11, float cameraProj_22,
    // Light Data
    int lightIndices[],
    int numLights,
    float light_positionView_x_array[],
    float light_positionView_y_array[],
    float light_positionView_z_array[],
    float light_attenuationEnd_array[],
    // Outputs
    int subtileIndices[],
    int subtileIndicesPitch,
    int subtileNumLights[]
    )
{
    float gBufferScale_x = 0.5f * (float)gBufferWidth;
    float gBufferScale_y = 0.5f * (float)gBufferHeight;

    float frustumPlanes_xy[2] = { -(cameraProj_11 * gBufferScale_x),
                                   (cameraProj_22 * gBufferScale_y) };
    float frustumPlanes_z[2] = { tileMidX - gBufferScale_x,
                                 tileMidY - gBufferScale_y };

    for (int i = 0; i < 2; ++i) {
        // Normalize
        float norm = 1.f / sqrtf(frustumPlanes_xy[i] * frustumPlanes_xy[i] +
                                 frustumPlanes_z[i] * frustumPlanes_z[i]);
        frustumPlanes_xy[i] *= norm;
        frustumPlanes_z[i] *= norm;
    }

    // Initialize
    int subtileLightOffset[4];
    subtileLightOffset[0] = 0 * subtileIndicesPitch;
    subtileLightOffset[1] = 1 * subtileIndicesPitch;
    subtileLightOffset[2] = 2 * subtileIndicesPitch;
    subtileLightOffset[3] = 3 * subtileIndicesPitch;

    for (int i = 0; i < numLights; ++i) {
        int lightIndex = lightIndices[i];

        float light_positionView_x = light_positionView_x_array[lightIndex];
        float light_positionView_y = light_positionView_y_array[lightIndex];
        float light_positionView_z = light_positionView_z_array[lightIndex];
        float light_attenuationEnd = light_attenuationEnd_array[lightIndex];
        float light_attenuationEndNeg = -light_attenuationEnd;

        // Test lights again against subtile z bounds
        bool inFrustum[4];
        inFrustum[0] = (light_positionView_z - subtileMinZ[0] >= light_attenuationEndNeg) &&
            (subtileMaxZ[0] - light_positionView_z >= light_attenuationEndNeg);
        inFrustum[1] = (light_positionView_z - subtileMinZ[1] >= light_attenuationEndNeg) &&
            (subtileMaxZ[1] - light_positionView_z >= light_attenuationEndNeg);
        inFrustum[2] = (light_positionView_z - subtileMinZ[2] >= light_attenuationEndNeg) &&
            (subtileMaxZ[2] - light_positionView_z >= light_attenuationEndNeg);
        inFrustum[3] = (light_positionView_z - subtileMinZ[3] >= light_attenuationEndNeg) &&
            (subtileMaxZ[3] - light_positionView_z >= light_attenuationEndNeg);

        float dx = light_positionView_z * frustumPlanes_z[0] +
            light_positionView_x * frustumPlanes_xy[0];
        float dy = light_positionView_z * frustumPlanes_z[1] +
            light_positionView_y * frustumPlanes_xy[1];

        if (fabsf(dx) > light_attenuationEnd) {
            bool positiveX = dx > 0.0f;
            inFrustum[0] = inFrustum[0] &&  positiveX;    // 00 subtile
            inFrustum[1] = inFrustum[1] && !positiveX;    // 10 subtile
            inFrustum[2] = inFrustum[2] &&  positiveX;    // 01 subtile
            inFrustum[3] = inFrustum[3] && !positiveX;    // 11 subtile
        }
        if (fabsf(dy) > light_attenuationEnd) {
            bool positiveY = dy > 0.0f;
            inFrustum[0] = inFrustum[0] &&  positiveY;    // 00 subtile
            inFrustum[1] = inFrustum[1] &&  positiveY;    // 10 subtile
            inFrustum[2] = inFrustum[2] && !positiveY;    // 01 subtile
            inFrustum[3] = inFrustum[3] && !positiveY;    // 11 subtile
        }

        if (inFrustum[0])
            subtileIndices[subtileLightOffset[0]++] = lightIndex;
        if (inFrustum[1])
            subtileIndices[subtileLightOffset[1]++] = lightIndex;
        if (inFrustum[2])
            subtileIndices[subtileLightOffset[2]++] = lightIndex;
        if (inFrustum[3])
            subtileIndices[subtileLightOffset[3]++] = lightIndex;
    }

    subtileNumLights[0] = subtileLightOffset[0] - 0 * subtileIndicesPitch;
    subtileNumLights[1] = subtileLightOffset[1] - 1 * subtileIndicesPitch;
    subtileNumLights[2] = subtileLightOffset[2] - 2 * subtileIndicesPitch;
    subtileNumLights[3] = subtileLightOffset[3] - 3 * subtileIndicesPitch;
}


static inline float
dot3(float x, float y, float z, float a, float b, float c) {
    return (x*a + y*b + z*c);
}


static inline void
normalize3(float x, float y, float z, float &ox, float &oy, float &oz) {
    float n = 1.f / sqrtf(x*x + y*y + z*z);
    ox = x * n;
    oy = y * n;
    oz = z * n;
}


static inline float
Unorm8ToFloat32(uint8_t u) {
    return (float)u * (1.0f / 255.0f);
}


static inline uint8_t
Float32ToUnorm8(float f) {
    return (uint8_t)(f * 255.0f);
}


static inline float
half_to_float_fast(uint16_t h) {
    uint32_t hs = h & (int32_t)0x8000u;  // Pick off sign bit
    uint32_t he = h & (int32_t)0x7C00u;  // Pick off exponent bits
    uint32_t hm = h & (int32_t)0x03FFu;  // Pick off mantissa bits

    // sign
    uint32_t xs = ((uint32_t) hs) << 16;
    // Exponent: unbias the halfp, then bias the single
    int32_t xes = ((int32_t) (he >> 10)) - 15 + 127;
    // Exponent
    uint32_t xe = (uint32_t) (xes << 23);
    // Mantissa
    uint32_t xm = ((uint32_t) hm) << 13;

    uint32_t bits = (xs | xe | xm);
    float *fp = reinterpret_cast<float *>(&bits);
    return *fp;
}


static void
ShadeTileC(
    int32_t tileStartX, int32_t tileEndX,
    int32_t tileStartY, int32_t tileEndY,
    int32_t gBufferWidth, int32_t gBufferHeight,
    const ispc::InputDataArrays &inputData,
    // Camera data
    float cameraProj_11, float cameraProj_22,
    float cameraProj_33, float cameraProj_43,
    // Light list
    int32_t tileLightIndices[],
    int32_t tileNumLights,
    // UI
    bool visualizeLightCount,
    // Output
    uint8_t framebuffer_r[],
    uint8_t framebuffer_g[],
    uint8_t framebuffer_b[]
    )
{
    if (tileNumLights == 0 || visualizeLightCount) {
        uint8_t c = (uint8_t)(std::min(tileNumLights << 2, 255));
        for (int32_t y = tileStartY; y < tileEndY; ++y) {
            for (int32_t x = tileStartX; x < tileEndX; ++x) {
                int32_t framebufferIndex = (y * gBufferWidth + x);
                framebuffer_r[framebufferIndex] = c;
                framebuffer_g[framebufferIndex] = c;
                framebuffer_b[framebufferIndex] = c;
            }
        }
    } else {
        float twoOverGBufferWidth = 2.0f / gBufferWidth;
        float twoOverGBufferHeight = 2.0f / gBufferHeight;

        for (int32_t y = tileStartY; y < tileEndY; ++y) {
            float positionScreen_y = -(((0.5f + y) * twoOverGBufferHeight) - 1.f);

            for (int32_t x = tileStartX; x < tileEndX; ++x) {
                int32_t gBufferOffset = y * gBufferWidth + x;

                // Reconstruct position and (negative) view vector from G-buffer
                float surface_positionView_x, surface_positionView_y, surface_positionView_z;
                float Vneg_x, Vneg_y, Vneg_z;

                float z = inputData.zBuffer[gBufferOffset];

                // Compute screen/clip-space position
                // NOTE: Mind DX11 viewport transform and pixel center!
                float positionScreen_x = (0.5f + (float)(x)) *
                    twoOverGBufferWidth - 1.0f;

                // Unproject depth buffer Z value into view space
                surface_positionView_z = cameraProj_43 / (z - cameraProj_33);
                surface_positionView_x = positionScreen_x * surface_positionView_z /
                    cameraProj_11;
                surface_positionView_y = positionScreen_y * surface_positionView_z /
                    cameraProj_22;

                // We actually end up with a vector pointing *at* the
                // surface (i.e. the negative view vector)
                normalize3(surface_positionView_x, surface_positionView_y,
                           surface_positionView_z, Vneg_x, Vneg_y, Vneg_z);

                // Reconstruct normal from G-buffer
                float surface_normal_x, surface_normal_y, surface_normal_z;
                float normal_x = half_to_float_fast(inputData.normalEncoded_x[gBufferOffset]);
                float normal_y = half_to_float_fast(inputData.normalEncoded_y[gBufferOffset]);

                float f = (normal_x - normal_x * normal_x) + (normal_y - normal_y * normal_y);
                float m = sqrtf(4.0f * f - 1.0f);

                surface_normal_x = m * (4.0f * normal_x - 2.0f);
                surface_normal_y = m * (4.0f * normal_y - 2.0f);
                surface_normal_z = 3.0f - 8.0f * f;

                // Load other G-buffer parameters
                float surface_specularAmount =
                    half_to_float_fast(inputData.specularAmount[gBufferOffset]);
                float surface_specularPower  =
                    half_to_float_fast(inputData.specularPower[gBufferOffset]);
                float surface_albedo_x = Unorm8ToFloat32(inputData.albedo_x[gBufferOffset]);
                float surface_albedo_y = Unorm8ToFloat32(inputData.albedo_y[gBufferOffset]);
                float surface_albedo_z = Unorm8ToFloat32(inputData.albedo_z[gBufferOffset]);

                float lit_x = 0.0f;
                float lit_y = 0.0f;
                float lit_z = 0.0f;
                for (int32_t tileLightIndex = 0; tileLightIndex < tileNumLights;
                     ++tileLightIndex) {
                    int32_t lightIndex = tileLightIndices[tileLightIndex];

                    // Gather light data relevant to initial culling
                    float light_positionView_x =
                        inputData.lightPositionView_x[lightIndex];
                    float light_positionView_y =
                        inputData.lightPositionView_y[lightIndex];
                    float light_positionView_z =
                        inputData.lightPositionView_z[lightIndex];
                    float light_attenuationEnd =
                        inputData.lightAttenuationEnd[lightIndex];

                    // Compute light vector
                    float L_x = light_positionView_x - surface_positionView_x;
                    float L_y = light_positionView_y - surface_positionView_y;
                    float L_z = light_positionView_z - surface_positionView_z;

                    float distanceToLight2 = dot3(L_x, L_y, L_z, L_x, L_y, L_z);

                    // Clip at end of attenuation
                    float light_attenutaionEnd2 = light_attenuationEnd * light_attenuationEnd;

                    if (distanceToLight2 < light_attenutaionEnd2) {
                        float distanceToLight = sqrtf(distanceToLight2);

                        float distanceToLightRcp = 1.f / distanceToLight;
                        L_x *= distanceToLightRcp;
                        L_y *= distanceToLightRcp;
                        L_z *= distanceToLightRcp;

                        // Start computing brdf
                        float NdotL = dot3(surface_normal_x, surface_normal_y,
                                           surface_normal_z, L_x, L_y, L_z);

                        // Clip back facing
                        if (NdotL > 0.0f) {
                            float light_attenuationBegin =
                                inputData.lightAttenuationBegin[lightIndex];

                            // Light distance attenuation (linstep)
                            float lightRange = (light_attenuationEnd - light_attenuationBegin);
                            float falloffPosition = (light_attenuationEnd - distanceToLight);
                            float attenuation = std::min(falloffPosition / lightRange, 1.0f);

                            float H_x = (L_x - Vneg_x);
                            float H_y = (L_y - Vneg_y);
                            float H_z = (L_z - Vneg_z);
                            normalize3(H_x, H_y, H_z, H_x, H_y, H_z);

                            float NdotH = dot3(surface_normal_x, surface_normal_y,
                                               surface_normal_z, H_x, H_y, H_z);
                            NdotH = std::max(NdotH, 0.0f);

                            float specular = powf(NdotH, surface_specularPower);
                            float specularNorm = (surface_specularPower + 2.0f) *
                                (1.0f / 8.0f);
                            float specularContrib = surface_specularAmount *
                                specularNorm * specular;

                            float k = attenuation * NdotL * (1.0f + specularContrib);

                            float light_color_x = inputData.lightColor_x[lightIndex];
                            float light_color_y = inputData.lightColor_y[lightIndex];
                            float light_color_z = inputData.lightColor_z[lightIndex];

                            float lightContrib_x = surface_albedo_x * light_color_x;
                            float lightContrib_y = surface_albedo_y * light_color_y;
                            float lightContrib_z = surface_albedo_z * light_color_z;

                            lit_x += lightContrib_x * k;
                            lit_y += lightContrib_y * k;
                            lit_z += lightContrib_z * k;
                        }
                    }
                }

                // Gamma correct
                float gamma = 1.0 / 2.2f;
                lit_x = powf(std::min(std::max(lit_x, 0.0f), 1.0f), gamma);
                lit_y = powf(std::min(std::max(lit_y, 0.0f), 1.0f), gamma);
                lit_z = powf(std::min(std::max(lit_z, 0.0f), 1.0f), gamma);

                framebuffer_r[gBufferOffset] = Float32ToUnorm8(lit_x);
                framebuffer_g[gBufferOffset] = Float32ToUnorm8(lit_y);
                framebuffer_b[gBufferOffset] = Float32ToUnorm8(lit_z);
            }
        }
    }
}


void
ShadeDynamicTileRecurse(InputData *input, int level, int tileX, int tileY,
                        int *lightIndices, int numLights,
                        Framebuffer *framebuffer) {
    const MinMaxZTree *minMaxZTree = gMinMaxZTree;

    // If we few enough lights or this is the base case (last level), shade
    // this full tile directly
    if (level == 0 || numLights < DYNAMIC_MIN_LIGHTS_TO_SUBDIVIDE) {
        int width = minMaxZTree->TileWidth(level);
        int height = minMaxZTree->TileHeight(level);
        int startX = tileX * width;
        int startY = tileY * height;
        int endX = std::min(input->header.framebufferWidth, startX + width);
        int endY = std::min(input->header.framebufferHeight, startY + height);

        // Skip entirely offscreen tiles
        if (endX > startX && endY > startY) {
            ShadeTileC(startX, endX, startY, endY,
                       input->header.framebufferWidth, input->header.framebufferHeight,
                       input->arrays,
                       input->header.cameraProj[0][0], input->header.cameraProj[1][1],
                       input->header.cameraProj[2][2], input->header.cameraProj[3][2],
                       lightIndices, numLights, VISUALIZE_LIGHT_COUNT,
                       framebuffer->r, framebuffer->g, framebuffer->b);
        }
    }
    else {
        // Otherwise, subdivide and 4-way recurse using X and Y splitting planes
        // Move down a level in the tree
        --level;
        tileX <<= 1;
        tileY <<= 1;
        int width = minMaxZTree->TileWidth(level);
        int height = minMaxZTree->TileHeight(level);

        // Work out splitting coords
        int midX = (tileX + 1) * width;
        int midY = (tileY + 1) * height;

        // Read subtile min/max data
        // NOTE: We must be sure to handle out-of-bounds access here since
        // sometimes we'll only have 1 or 2 subtiles for non-pow-2
        // framebuffer sizes.
        bool rightTileExists = (tileX + 1 < minMaxZTree->NumTilesX(level));
        bool bottomTileExists = (tileY + 1 < minMaxZTree->NumTilesY(level));

        // NOTE: Order is 00, 10, 01, 11
        // Set defaults up to cull all lights if the tile doesn't exist (offscreen)
        float minZ[4] = {input->header.cameraFar, input->header.cameraFar,
                         input->header.cameraFar, input->header.cameraFar};
        float maxZ[4] = {input->header.cameraNear, input->header.cameraNear,
                         input->header.cameraNear, input->header.cameraNear};

        minZ[0] = minMaxZTree->MinZ(level, tileX, tileY);
        maxZ[0] = minMaxZTree->MaxZ(level, tileX, tileY);
        if (rightTileExists) {
            minZ[1] = minMaxZTree->MinZ(level, tileX + 1, tileY);
            maxZ[1] = minMaxZTree->MaxZ(level, tileX + 1, tileY);
            if (bottomTileExists) {
                minZ[3] = minMaxZTree->MinZ(level, tileX + 1, tileY + 1);
                maxZ[3] = minMaxZTree->MaxZ(level, tileX + 1, tileY + 1);
            }
        }
        if (bottomTileExists) {
            minZ[2] = minMaxZTree->MinZ(level, tileX, tileY + 1);
            maxZ[2] = minMaxZTree->MaxZ(level, tileX, tileY + 1);
        }

        // Cull lights into subtile lists
#ifdef ISPC_IS_WINDOWS
        __declspec(align(ALIGNMENT_BYTES))
#endif
            int subtileLightIndices[4][MAX_LIGHTS]
#ifndef ISPC_IS_WINDOWS
            __attribute__ ((aligned(ALIGNMENT_BYTES)))
#endif
;
        int subtileNumLights[4];
        SplitTileMinMax(midX, midY, minZ, maxZ,
            input->header.framebufferWidth, input->header.framebufferHeight,
            input->header.cameraProj[0][0], input->header.cameraProj[1][1],
            lightIndices, numLights, input->arrays.lightPositionView_x,
            input->arrays.lightPositionView_y, input->arrays.lightPositionView_z,
            input->arrays.lightAttenuationEnd,
            subtileLightIndices[0], MAX_LIGHTS, subtileNumLights);

        // Recurse into subtiles
        ShadeDynamicTileRecurse(input, level, tileX    , tileY,
                                subtileLightIndices[0], subtileNumLights[0],
                                framebuffer);
        ShadeDynamicTileRecurse(input, level, tileX + 1, tileY,
                                subtileLightIndices[1], subtileNumLights[1],
                                framebuffer);
        ShadeDynamicTileRecurse(input, level, tileX    , tileY + 1,
                                subtileLightIndices[2], subtileNumLights[2],
                                framebuffer);
        ShadeDynamicTileRecurse(input, level, tileX + 1, tileY + 1,
                                subtileLightIndices[3], subtileNumLights[3],
                                framebuffer);
    }
}


static int
IntersectLightsWithTileMinMax(
    int tileStartX, int tileEndX,
    int tileStartY, int tileEndY,
    // Tile data
    float minZ,
    float maxZ,
    // G-buffer data
    int gBufferWidth, int gBufferHeight,
    // Camera data
    float cameraProj_11, float cameraProj_22,
    // Light Data
    int numLights,
    float light_positionView_x_array[],
    float light_positionView_y_array[],
    float light_positionView_z_array[],
    float light_attenuationEnd_array[],
    // Output
    int tileLightIndices[]
    )
{
    float gBufferScale_x = 0.5f * (float)gBufferWidth;
    float gBufferScale_y = 0.5f * (float)gBufferHeight;

    float frustumPlanes_xy[4];
    float frustumPlanes_z[4];

    // This one is totally constant over the whole screen... worth pulling it up at all?
    float frustumPlanes_xy_v[4] = { -(cameraProj_11 * gBufferScale_x),
                                    (cameraProj_11 * gBufferScale_x),
                                    (cameraProj_22 * gBufferScale_y),
                                    -(cameraProj_22 * gBufferScale_y) };

    float frustumPlanes_z_v[4] = {  tileEndX - gBufferScale_x,
                                    -tileStartX + gBufferScale_x,
                                    tileEndY - gBufferScale_y,
                                    -tileStartY + gBufferScale_y };

    for (int i = 0; i < 4; ++i) {
        float norm = 1.f / sqrtf(frustumPlanes_xy_v[i] * frustumPlanes_xy_v[i] +
                                 frustumPlanes_z_v[i] * frustumPlanes_z_v[i]);
        frustumPlanes_xy_v[i] *= norm;
        frustumPlanes_z_v[i] *= norm;

        frustumPlanes_xy[i] = frustumPlanes_xy_v[i];
        frustumPlanes_z[i] = frustumPlanes_z_v[i];
    }

    int tileNumLights = 0;

    for (int lightIndex = 0; lightIndex < numLights; ++lightIndex) {
        float light_positionView_z = light_positionView_z_array[lightIndex];
        float light_attenuationEnd = light_attenuationEnd_array[lightIndex];
        float light_attenuationEndNeg = -light_attenuationEnd;

        float d = light_positionView_z - minZ;
        bool inFrustum = (d >= light_attenuationEndNeg);

        d = maxZ - light_positionView_z;
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);

        if (!inFrustum)
            continue;

        float light_positionView_x = light_positionView_x_array[lightIndex];
        float light_positionView_y = light_positionView_y_array[lightIndex];

        d = light_positionView_z * frustumPlanes_z[0] +
            light_positionView_x * frustumPlanes_xy[0];
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);

        d = light_positionView_z * frustumPlanes_z[1] +
            light_positionView_x * frustumPlanes_xy[1];
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);

        d = light_positionView_z * frustumPlanes_z[2] +
            light_positionView_y * frustumPlanes_xy[2];
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);

        d = light_positionView_z * frustumPlanes_z[3] +
            light_positionView_y * frustumPlanes_xy[3];
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);

        // Pack and store intersecting lights
        if (inFrustum)
            tileLightIndices[tileNumLights++] = lightIndex;
    }

    return tileNumLights;
}


void
ShadeDynamicTile(InputData *input, int level, int tileX, int tileY,
                 Framebuffer *framebuffer) {
    const MinMaxZTree *minMaxZTree = gMinMaxZTree;

    // Get Z min/max for this tile
    int width = minMaxZTree->TileWidth(level);
    int height = minMaxZTree->TileHeight(level);
    float minZ = minMaxZTree->MinZ(level, tileX, tileY);
    float maxZ = minMaxZTree->MaxZ(level, tileX, tileY);

    int startX = tileX * width;
    int startY = tileY * height;
    int endX = std::min(input->header.framebufferWidth, startX + width);
    int endY = std::min(input->header.framebufferHeight, startY + height);

    // This is a root tile, so first do a full 6-plane cull
#ifdef ISPC_IS_WINDOWS
    __declspec(align(ALIGNMENT_BYTES))
#endif
        int lightIndices[MAX_LIGHTS]
#ifndef ISPC_IS_WINDOWS
        __attribute__ ((aligned(ALIGNMENT_BYTES)))
#endif
;
    int numLights = IntersectLightsWithTileMinMax(
        startX, endX, startY, endY,    minZ, maxZ,
        input->header.framebufferWidth, input->header.framebufferHeight,
        input->header.cameraProj[0][0], input->header.cameraProj[1][1],
        MAX_LIGHTS, input->arrays.lightPositionView_x,
        input->arrays.lightPositionView_y, input->arrays.lightPositionView_z,
        input->arrays.lightAttenuationEnd, lightIndices);

    // Now kick off the recursive process for this tile
    ShadeDynamicTileRecurse(input, level, tileX, tileY, lightIndices,
                            numLights, framebuffer);
}


void
DispatchDynamicC(InputData *input, Framebuffer *framebuffer)
{
    MinMaxZTree *minMaxZTree = gMinMaxZTree;

    // Update min/max Z tree
    minMaxZTree->Update(input->arrays.zBuffer, input->header.framebufferWidth,
        input->header.cameraProj[2][2], input->header.cameraProj[3][2],
        input->header.cameraNear, input->header.cameraFar);

    int rootLevel = minMaxZTree->Levels() - 1;
    int rootTilesX = minMaxZTree->NumTilesX(rootLevel);
    int rootTilesY = minMaxZTree->NumTilesY(rootLevel);
    int rootTiles = rootTilesX * rootTilesY;
    for (int g = 0; g < rootTiles; ++g) {
        uint32_t tileY = g / rootTilesX;
        uint32_t tileX = g % rootTilesX;
        ShadeDynamicTile(input, rootLevel, tileX, tileY, framebuffer);
    }
}
