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

#ifdef __NVPTX__
#warning "emitting DEVICE code"
#define programCount warpSize()
#define programIndex laneIndex()
#define taskIndex    blockIndex0()
#define taskCount    blockCount0()
#define cif          if
#else
#warning "emitting HOST code"
#endif


#include "deferred.h"

struct InputDataArrays
{
    float *zBuffer;
    unsigned int16 *normalEncoded_x; // half float
    unsigned int16 *normalEncoded_y; // half float
    unsigned int16 *specularAmount; // half float
    unsigned int16 *specularPower; // half float
    unsigned int8 *albedo_x; // unorm8
    unsigned int8 *albedo_y; // unorm8
    unsigned int8 *albedo_z; // unorm8
    float *lightPositionView_x;
    float *lightPositionView_y;
    float *lightPositionView_z;
    float *lightAttenuationBegin;
    float *lightColor_x;
    float *lightColor_y;
    float *lightColor_z;
    float *lightAttenuationEnd;
};

struct InputHeader
{
    float cameraProj[4][4];
    float cameraNear;
    float cameraFar;

    int32 framebufferWidth;
    int32 framebufferHeight;
    int32 numLights;
    int32 inputDataChunkSize;
    int32 inputDataArrayOffsets[idaNum];
};


///////////////////////////////////////////////////////////////////////////
// Common utility routines

static inline float
dot3(float x, float y, float z, float a, float b, float c) {
    return (x*a + y*b + z*c);
}


static inline void
normalize3(float x, float y, float z, float &ox, float &oy, float &oz) {
    float n = rsqrt(x*x + y*y + z*z);
    ox = x * n;
    oy = y * n;
    oz = z * n;
}


static inline float
Unorm8ToFloat32(unsigned int8 u) {
    return (float)u * (1.0f / 255.0f);
}


static inline unsigned int8
Float32ToUnorm8(float f) {
    return (unsigned int8)(f * 255.0f);
}


static inline void
ComputeZBounds(
    uniform int32 tileStartX, uniform int32 tileEndX,
    uniform int32 tileStartY, uniform int32 tileEndY,
    // G-buffer data
    uniform float zBuffer[],
    uniform int32 gBufferWidth,
    // Camera data
    uniform float cameraProj_33, uniform float cameraProj_43,
    uniform float cameraNear, uniform float cameraFar,
    // Output
    uniform float &minZ,
    uniform float &maxZ
    )
{
    // Find Z bounds
    float laneMinZ = cameraFar;
    float laneMaxZ = cameraNear;
    for (uniform int32 y = tileStartY; y < tileEndY; ++y) 
      foreach (x = tileStartX ... tileEndX) 
      {
        // Unproject depth buffer Z value into view space
        float z = zBuffer[y * gBufferWidth + x];
        float viewSpaceZ = cameraProj_43 / (z - cameraProj_33);

        // Work out Z bounds for our samples
        // Avoid considering skybox/background or otherwise invalid pixels
        if ((viewSpaceZ < cameraFar) && (viewSpaceZ >= cameraNear)) {
          laneMinZ = min(laneMinZ, viewSpaceZ);
          laneMaxZ = max(laneMaxZ, viewSpaceZ);
        }
      }
    minZ = reduce_min(laneMinZ);
    maxZ = reduce_max(laneMaxZ);
}


static inline uniform int32
IntersectLightsWithTileMinMax(
    uniform int32 tileStartX, uniform int32 tileEndX,
    uniform int32 tileStartY, uniform int32 tileEndY,
    // Tile data
    uniform float minZ,
    uniform float maxZ,
    // G-buffer data
    uniform int32 gBufferWidth, uniform int32 gBufferHeight,
    // Camera data
    uniform float cameraProj_11, uniform float cameraProj_22,
    // Light Data
    uniform int32 numLights,
    uniform float light_positionView_x_array[],
    uniform float light_positionView_y_array[],
    uniform float light_positionView_z_array[],
    uniform float light_attenuationEnd_array[],
    // Output
    uniform int32 tileLightIndices[]
    )
{
    uniform float gBufferScale_x = 0.5f * (float)gBufferWidth;
    uniform float gBufferScale_y = 0.5f * (float)gBufferHeight;
        
    uniform float frustumPlanes_xy[4] = {
        -(cameraProj_11 * gBufferScale_x),
         (cameraProj_11 * gBufferScale_x),
         (cameraProj_22 * gBufferScale_y),
        -(cameraProj_22 * gBufferScale_y) };
    uniform float frustumPlanes_z[4] = {
         tileEndX - gBufferScale_x,
        -tileStartX + gBufferScale_x,
         tileEndY - gBufferScale_y,
        -tileStartY + gBufferScale_y };

    for (uniform int i = 0; i < 4; ++i) {
        uniform float norm = rsqrt(frustumPlanes_xy[i] * frustumPlanes_xy[i] + 
                                   frustumPlanes_z[i] * frustumPlanes_z[i]);
        frustumPlanes_xy[i] *= norm;
        frustumPlanes_z[i] *= norm;
    }

    uniform int32 tileNumLights = 0;

    foreach (lightIndex = 0 ... numLights) 
    {
      float light_positionView_z = light_positionView_z_array[lightIndex];
      float light_attenuationEnd = light_attenuationEnd_array[lightIndex];
      float light_attenuationEndNeg = -light_attenuationEnd;

      float d = light_positionView_z - minZ;
      bool inFrustum = (d >= light_attenuationEndNeg);

      d = maxZ - light_positionView_z;
      inFrustum = inFrustum && (d >= light_attenuationEndNeg);

      // This seems better than cif(!inFrustum) ccontinue; here since we
      // don't actually need to mask the rest of this function - this is
      // just a greedy early-out.  Could also structure all of this as
      // nested if() statements, but this a bit easier to read
      if (any(inFrustum)) 
      {
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
        const bool active = inFrustum && lightIndex < numLights;

        if(any(active))
          tileNumLights += packed_store_active(active, &tileLightIndices[tileNumLights], lightIndex);
      }
    }

    return tileNumLights;
}


static inline  uniform int32
IntersectLightsWithTile(
    uniform int32 tileStartX, uniform int32 tileEndX,
    uniform int32 tileStartY, uniform int32 tileEndY,
    uniform int32 gBufferWidth, uniform int32 gBufferHeight,
    // G-buffer data
    uniform float zBuffer[],
    // Camera data
    uniform float cameraProj_11, uniform float cameraProj_22,
    uniform float cameraProj_33, uniform float cameraProj_43,
    uniform float cameraNear, uniform float cameraFar,
    // Light Data
    uniform int32 numLights,
    uniform float light_positionView_x_array[],
    uniform float light_positionView_y_array[],
    uniform float light_positionView_z_array[],
    uniform float light_attenuationEnd_array[],
    // Output
    uniform int32 tileLightIndices[]
    )
{
    uniform float minZ, maxZ;
    ComputeZBounds(tileStartX, tileEndX, tileStartY, tileEndY,
        zBuffer, gBufferWidth, cameraProj_33, cameraProj_43, cameraNear, cameraFar,
        minZ, maxZ);

    uniform int32 tileNumLights = IntersectLightsWithTileMinMax(
        tileStartX, tileEndX, tileStartY, tileEndY, minZ, maxZ,
        gBufferWidth, gBufferHeight, cameraProj_11, cameraProj_22,
        MAX_LIGHTS, light_positionView_x_array, light_positionView_y_array, 
        light_positionView_z_array, light_attenuationEnd_array,
        tileLightIndices);

    return tileNumLights;
}


static inline void
ShadeTile(
    uniform int32 tileStartX, uniform int32 tileEndX,
    uniform int32 tileStartY, uniform int32 tileEndY,
    uniform int32 gBufferWidth, uniform int32 gBufferHeight,
    const uniform InputDataArrays &inputData,
    // Camera data
    uniform float cameraProj_11, uniform float cameraProj_22,
    uniform float cameraProj_33, uniform float cameraProj_43,
    // Light list
    uniform int32 tileLightIndices[],
    uniform int32 tileNumLights,
    // UI
    uniform bool visualizeLightCount,
    // Output
    uniform unsigned int8 framebuffer_r[],
    uniform unsigned int8 framebuffer_g[],
    uniform unsigned int8 framebuffer_b[]
    )
{
    if (tileNumLights == 0 || visualizeLightCount) {
        uniform unsigned int8 c = (unsigned int8)(min(tileNumLights << 2, 255));
        for (uniform int32 y = tileStartY; y < tileEndY; ++y) 
          foreach (x = tileStartX ... tileEndX) 
          { 
            int32 framebufferIndex = (y * gBufferWidth + x);
            framebuffer_r[framebufferIndex] = c;
            framebuffer_g[framebufferIndex] = c;
            framebuffer_b[framebufferIndex] = c;
          }
    } else {
        uniform float twoOverGBufferWidth = 2.0f / gBufferWidth;
        uniform float twoOverGBufferHeight = 2.0f / gBufferHeight;
        
        for (uniform int32 y = tileStartY; y < tileEndY; ++y) {
          uniform float positionScreen_y = -(((0.5f + y) * twoOverGBufferHeight) - 1.f);

          foreach (x = tileStartX ... tileEndX) {
            int32 gBufferOffset = y * gBufferWidth + x;

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
            float normal_x = half_to_float(inputData.normalEncoded_x[gBufferOffset]);
            float normal_y = half_to_float(inputData.normalEncoded_y[gBufferOffset]);

            float f = (normal_x - normal_x * normal_x) + (normal_y - normal_y * normal_y);
            float m = sqrt(4.0f * f - 1.0f);

            surface_normal_x = m * (4.0f * normal_x - 2.0f);
            surface_normal_y = m * (4.0f * normal_y - 2.0f);
            surface_normal_z = 3.0f - 8.0f * f;

            // Load other G-buffer parameters
            float surface_specularAmount = 
              half_to_float(inputData.specularAmount[gBufferOffset]);
            float surface_specularPower  = 
              half_to_float(inputData.specularPower[gBufferOffset]);
            float surface_albedo_x = Unorm8ToFloat32(inputData.albedo_x[gBufferOffset]);
            float surface_albedo_y = Unorm8ToFloat32(inputData.albedo_y[gBufferOffset]);
            float surface_albedo_z = Unorm8ToFloat32(inputData.albedo_z[gBufferOffset]);

            float lit_x = 0.0f;
            float lit_y = 0.0f;
            float lit_z = 0.0f;
            for (uniform int32 tileLightIndex = 0; tileLightIndex < tileNumLights; 
                ++tileLightIndex) {
              uniform int32 lightIndex = tileLightIndices[tileLightIndex];

              // Gather light data relevant to initial culling
              uniform float light_positionView_x = 
                inputData.lightPositionView_x[lightIndex];
              uniform float light_positionView_y = 
                inputData.lightPositionView_y[lightIndex];
              uniform float light_positionView_z = 
                inputData.lightPositionView_z[lightIndex];
              uniform float light_attenuationEnd = 
                inputData.lightAttenuationEnd[lightIndex];

              // Compute light vector
              float L_x = light_positionView_x - surface_positionView_x;
              float L_y = light_positionView_y - surface_positionView_y;
              float L_z = light_positionView_z - surface_positionView_z;

              float distanceToLight2 = dot3(L_x, L_y, L_z, L_x, L_y, L_z);

              // Clip at end of attenuation
              float light_attenutaionEnd2 = light_attenuationEnd * light_attenuationEnd;

              cif (distanceToLight2 < light_attenutaionEnd2) {                    
                float distanceToLight = sqrt(distanceToLight2);

                // HLSL "rcp" is allowed to be fairly inaccurate
                float distanceToLightRcp = rcp(distanceToLight);
                L_x *= distanceToLightRcp;
                L_y *= distanceToLightRcp;
                L_z *= distanceToLightRcp;

                // Start computing brdf
                float NdotL = dot3(surface_normal_x, surface_normal_y, 
                    surface_normal_z, L_x, L_y, L_z);

                // Clip back facing
                cif (NdotL > 0.0f) {
                  uniform float light_attenuationBegin = 
                    inputData.lightAttenuationBegin[lightIndex];

                  // Light distance attenuation (linstep)
                  float lightRange = (light_attenuationEnd - light_attenuationBegin);
                  float falloffPosition = (light_attenuationEnd - distanceToLight);
                  float attenuation = min(falloffPosition / lightRange, 1.0f);

                  float H_x = (L_x - Vneg_x);
                  float H_y = (L_y - Vneg_y);
                  float H_z = (L_z - Vneg_z);
                  normalize3(H_x, H_y, H_z, H_x, H_y, H_z);

                  float NdotH = dot3(surface_normal_x, surface_normal_y, 
                      surface_normal_z, H_x, H_y, H_z);
                  NdotH = max(NdotH, 0.0f);

                  float specular = pow(NdotH, surface_specularPower);
                  float specularNorm = (surface_specularPower + 2.0f) * 
                    (1.0f / 8.0f);
                  float specularContrib = surface_specularAmount * 
                    specularNorm * specular;

                  float k = attenuation * NdotL * (1.0f + specularContrib);

                  uniform float light_color_x = inputData.lightColor_x[lightIndex];
                  uniform float light_color_y = inputData.lightColor_y[lightIndex];
                  uniform float light_color_z = inputData.lightColor_z[lightIndex];

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
            // These pows are pretty slow right now, but we can do
            // something faster if really necessary to squeeze every
            // last bit of performance out of it
            float gamma = 1.0 / 2.2f;
            lit_x = pow(clamp(lit_x, 0.0f, 1.0f), gamma);
            lit_y = pow(clamp(lit_y, 0.0f, 1.0f), gamma);
            lit_z = pow(clamp(lit_z, 0.0f, 1.0f), gamma);

            framebuffer_r[gBufferOffset] = Float32ToUnorm8(lit_x);
            framebuffer_g[gBufferOffset] = Float32ToUnorm8(lit_y);
            framebuffer_b[gBufferOffset] = Float32ToUnorm8(lit_z);
          }
        }
    }
}


///////////////////////////////////////////////////////////////////////////
// Static decomposition

void task
RenderTile(uniform int num_groups_x, uniform int num_groups_y,
           const  uniform InputHeader inputHeaderPtr[],
           const  uniform InputDataArrays inputDataPtr[],
           uniform int visualizeLightCount,
           // Output
           uniform unsigned int8 framebuffer_r[],
           uniform unsigned int8 framebuffer_g[],
           uniform unsigned int8 framebuffer_b[]) {
  if (taskIndex >= taskCount) return;
  const  uniform InputHeader inputHeader = *inputHeaderPtr;
  const  uniform InputDataArrays inputData = *inputDataPtr;

    uniform int32 group_y = taskIndex / num_groups_x;
    uniform int32 group_x = taskIndex % num_groups_x;
    uniform int32 tile_start_x = group_x * MIN_TILE_WIDTH;
    uniform int32 tile_start_y = group_y * MIN_TILE_HEIGHT;
    uniform int32 tile_end_x = tile_start_x + MIN_TILE_WIDTH;
    uniform int32 tile_end_y = tile_start_y + MIN_TILE_HEIGHT;

    uniform int framebufferWidth = inputHeader.framebufferWidth;
    uniform int framebufferHeight = inputHeader.framebufferHeight;
    uniform float cameraProj_00 = inputHeader.cameraProj[0][0];
    uniform float cameraProj_11 = inputHeader.cameraProj[1][1];
    uniform float cameraProj_22 = inputHeader.cameraProj[2][2];
    uniform float cameraProj_32 = inputHeader.cameraProj[3][2];

    // Light intersection: figure out which lights illuminate this tile.
#if 1
    uniform int * uniform tileLightIndices = uniform new uniform int [MAX_LIGHTS];
#else
    uniform int tileLightIndices[MAX_LIGHTS];  // Light list for the tile
#endif
#if 1
    uniform int numTileLights = 
        IntersectLightsWithTile(tile_start_x, tile_end_x, 
                                tile_start_y, tile_end_y,
                                framebufferWidth, framebufferHeight,
                                inputData.zBuffer,
                                cameraProj_00, cameraProj_11,
                                cameraProj_22, cameraProj_32,
                                inputHeader.cameraNear, inputHeader.cameraFar,
                                MAX_LIGHTS,
                                inputData.lightPositionView_x, 
                                inputData.lightPositionView_y, 
                                inputData.lightPositionView_z, 
                                inputData.lightAttenuationEnd,
                                tileLightIndices);

    // And now shade the tile, using the lights in tileLightIndices
    ShadeTile(tile_start_x, tile_end_x, tile_start_y, tile_end_y,
              framebufferWidth, framebufferHeight, inputData,
              cameraProj_00, cameraProj_11, cameraProj_22, cameraProj_32,
              tileLightIndices, numTileLights, visualizeLightCount, 
              framebuffer_r, framebuffer_g, framebuffer_b);
#endif
#if 1
    delete tileLightIndices;
#endif
}


export void
RenderStatic(uniform InputHeader inputHeaderPtr[],
             uniform InputDataArrays inputDataPtr[],
             uniform int visualizeLightCount,
             // Output
             uniform unsigned int8 framebuffer_r[],
             uniform unsigned int8 framebuffer_g[],
             uniform unsigned int8 framebuffer_b[]) {

  const uniform InputHeader inputHeader = *inputHeaderPtr;
  const uniform InputDataArrays inputData = *inputDataPtr;


    uniform int num_groups_x = (inputHeader.framebufferWidth + 
                                MIN_TILE_WIDTH - 1) / MIN_TILE_WIDTH;
    uniform int num_groups_y = (inputHeader.framebufferHeight + 
                                MIN_TILE_HEIGHT - 1) / MIN_TILE_HEIGHT;
    uniform int num_groups = num_groups_x * num_groups_y;

    // Launch a task to render each tile, each of which is MIN_TILE_WIDTH
    // by MIN_TILE_HEIGHT pixels.
    launch[num_groups] RenderTile(num_groups_x, num_groups_y,
                                  inputHeaderPtr, inputDataPtr, visualizeLightCount,
                                  framebuffer_r, framebuffer_g, framebuffer_b);
}



