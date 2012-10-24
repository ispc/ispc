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

#ifdef __cilk

#include "deferred.h"
#include "kernels_ispc.h"
#include <algorithm>
#include <assert.h>

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
#define MIN_TILE_WIDTH 16
#define MIN_TILE_HEIGHT 16


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


class MinMaxZTreeCilk
{
public:
    // Currently (min) tile dimensions must divide gBuffer dimensions evenly
    // Levels must be small enough that neither dimension goes below one tile
    MinMaxZTreeCilk(
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
        // Compute level 0 in parallel. Outer loops is here since we use Cilk
        _Cilk_for (int tileY = 0; tileY < mNumTilesY; ++tileY) {
            ispc::ComputeZBoundsRow(tileY,
                mTileWidth, mTileHeight, mNumTilesX, mNumTilesY,
                zBuffer, gBufferPitchInElements,
                cameraProj_33, cameraProj_43, cameraNear, cameraFar,
                mMinZArrays[0] + (tileY * mNumTilesX),
                mMaxZArrays[0] + (tileY * mNumTilesX));
        }

        // Generate other levels
        // NOTE: We currently don't use ispc here since it's sort of an
        // awkward gather-based reduction Using SSE odd pack/unpack
        // instructions might actually work here when we need to optimize
        for (int level = 1; level < mLevels; ++level) {
            int destTilesX = NumTilesX(level);
            int destTilesY = NumTilesY(level);
            int srcLevel = level - 1;
            int srcTilesX = NumTilesX(srcLevel);
            int srcTilesY = NumTilesY(srcLevel);
            _Cilk_for (int y = 0; y < destTilesY; ++y) {
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

    ~MinMaxZTreeCilk() {
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

static MinMaxZTreeCilk *gMinMaxZTreeCilk = 0;

void InitDynamicCilk(InputData *input) {
    gMinMaxZTreeCilk = 
        new MinMaxZTreeCilk(MIN_TILE_WIDTH, MIN_TILE_HEIGHT, DYNAMIC_TREE_LEVELS,
                            input->header.framebufferWidth, 
                            input->header.framebufferHeight);
}


static void
ShadeDynamicTileRecurse(InputData *input, int level, int tileX, int tileY, 
                        int *lightIndices, int numLights, 
                        Framebuffer *framebuffer) {
    const MinMaxZTreeCilk *minMaxZTree = gMinMaxZTreeCilk;
    
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
            ispc::ShadeTile(
                startX, endX, startY, endY,
                input->header.framebufferWidth, input->header.framebufferHeight,
                &input->arrays,
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
        ispc::SplitTileMinMax(midX, midY, minZ, maxZ,
            input->header.framebufferWidth, input->header.framebufferHeight, 
            input->header.cameraProj[0][0], input->header.cameraProj[1][1],
            lightIndices, numLights, input->arrays.lightPositionView_x, 
            input->arrays.lightPositionView_y, input->arrays.lightPositionView_z, 
            input->arrays.lightAttenuationEnd,
            subtileLightIndices[0], MAX_LIGHTS, subtileNumLights);
        
        // Recurse into subtiles
        _Cilk_spawn ShadeDynamicTileRecurse(input, level, tileX    , tileY, 
                                            subtileLightIndices[0], subtileNumLights[0],
                                            framebuffer);
        _Cilk_spawn ShadeDynamicTileRecurse(input, level, tileX + 1, tileY,
                                            subtileLightIndices[1], subtileNumLights[1],
                                            framebuffer);
        _Cilk_spawn ShadeDynamicTileRecurse(input, level, tileX    , tileY + 1,
                                            subtileLightIndices[2], subtileNumLights[2],
                                            framebuffer);
        ShadeDynamicTileRecurse(input, level, tileX + 1, tileY + 1,
                                subtileLightIndices[3], subtileNumLights[3],
                                framebuffer);
    }
}


static void
ShadeDynamicTile(InputData *input, int level, int tileX, int tileY,
                 Framebuffer *framebuffer) {
    const MinMaxZTreeCilk *minMaxZTree = gMinMaxZTreeCilk;

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
    int numLights = ispc::IntersectLightsWithTileMinMax(
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
DispatchDynamicCilk(InputData *input, Framebuffer *framebuffer)
{
    MinMaxZTreeCilk *minMaxZTree = gMinMaxZTreeCilk;
        
    // Update min/max Z tree
    minMaxZTree->Update(input->arrays.zBuffer, input->header.framebufferWidth,
        input->header.cameraProj[2][2], input->header.cameraProj[3][2], 
        input->header.cameraNear, input->header.cameraFar);

    // Launch the "root" tiles.  Ideally these should at least fill the
    // machine... at the moment we have a static number of "levels" to the
    // mip tree but it might make sense to compute it based on the width of
    // the machine.
    int rootLevel = minMaxZTree->Levels() - 1;
    int rootTilesX = minMaxZTree->NumTilesX(rootLevel);
    int rootTilesY = minMaxZTree->NumTilesY(rootLevel);
    int rootTiles = rootTilesX * rootTilesY;
    _Cilk_for (int g = 0; g < rootTiles; ++g) {
        uint32_t tileY = g / rootTilesX;
        uint32_t tileX = g % rootTilesX;
        ShadeDynamicTile(input, rootLevel, tileX, tileY, framebuffer);
    }
}

#endif // __cilk
