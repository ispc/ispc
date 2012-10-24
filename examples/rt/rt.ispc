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

#define bool int

typedef float<3> float3;

struct Ray {
    float3 origin, dir, invDir;
    uniform unsigned int dirIsNeg[3];
    float mint, maxt;
    int hitId;
};

struct Triangle {
    float p[3][4];
    int id;
    int pad[3];
};

struct LinearBVHNode {
    float bounds[2][3];
    unsigned int offset;     // num primitives for leaf, second child for interior
    unsigned int8 nPrimitives;
    unsigned int8 splitAxis;
    unsigned int16 pad;
};

static inline float3 Cross(const float3 v1, const float3 v2) {
    float v1x = v1.x, v1y = v1.y, v1z = v1.z;
    float v2x = v2.x, v2y = v2.y, v2z = v2.z;
    float3 ret;
    ret.x = (v1y * v2z) - (v1z * v2y);
    ret.y = (v1z * v2x) - (v1x * v2z);
    ret.z = (v1x * v2y) - (v1y * v2x);
    return ret;
}

static inline float Dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


static void generateRay(uniform const float raster2camera[4][4], 
                        uniform const float camera2world[4][4],
                        float x, float y, Ray &ray) {
    ray.mint = 0.f;
    ray.maxt = 1e30f;

    ray.hitId = 0;

    // transform raster coordinate (x, y, 0) to camera space
    float camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
    float camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
    float camz = raster2camera[2][3];
    float camw = raster2camera[3][3];
    camx /= camw;
    camy /= camw;
    camz /= camw;

    ray.dir.x = camera2world[0][0] * camx + camera2world[0][1] * camy + 
        camera2world[0][2] * camz;
    ray.dir.y = camera2world[1][0] * camx + camera2world[1][1] * camy + 
        camera2world[1][2] * camz;
    ray.dir.z = camera2world[2][0] * camx + camera2world[2][1] * camy + 
        camera2world[2][2] * camz;

    ray.origin.x = camera2world[0][3] / camera2world[3][3];
    ray.origin.y = camera2world[1][3] / camera2world[3][3];
    ray.origin.z = camera2world[2][3] / camera2world[3][3];

    ray.invDir = 1.f / ray.dir;

    ray.dirIsNeg[0] = any(ray.invDir.x < 0) ? 1 : 0;
    ray.dirIsNeg[1] = any(ray.invDir.y < 0) ? 1 : 0;
    ray.dirIsNeg[2] = any(ray.invDir.z < 0) ? 1 : 0;
}


static bool BBoxIntersect(const uniform float bounds[2][3], 
                          const Ray &ray) {
    uniform float3 bounds0 = { bounds[0][0], bounds[0][1], bounds[0][2] };
    uniform float3 bounds1 = { bounds[1][0], bounds[1][1], bounds[1][2] };
    float t0 = ray.mint, t1 = ray.maxt;

    // Check all three axis-aligned slabs.  Don't try to early out; it's
    // not worth the trouble
    float3 tNear = (bounds0 - ray.origin) * ray.invDir;
    float3 tFar  = (bounds1 - ray.origin) * ray.invDir;
    if (tNear.x > tFar.x) {
        float tmp = tNear.x;
        tNear.x = tFar.x;
        tFar.x = tmp;
    }
    t0 = max(tNear.x, t0);
    t1 = min(tFar.x, t1);

    if (tNear.y > tFar.y) {
        float tmp = tNear.y;
        tNear.y = tFar.y;
        tFar.y = tmp;
    }
    t0 = max(tNear.y, t0);
    t1 = min(tFar.y, t1);

    if (tNear.z > tFar.z) {
        float tmp = tNear.z;
        tNear.z = tFar.z;
        tFar.z = tmp;
    }
    t0 = max(tNear.z, t0);
    t1 = min(tFar.z, t1);
    
    return (t0 <= t1);
}



static bool TriIntersect(const uniform Triangle &tri, Ray &ray) {
    uniform float3 p0 = { tri.p[0][0], tri.p[0][1], tri.p[0][2] };
    uniform float3 p1 = { tri.p[1][0], tri.p[1][1], tri.p[1][2] };
    uniform float3 p2 = { tri.p[2][0], tri.p[2][1], tri.p[2][2] };
    uniform float3 e1 = p1 - p0;
    uniform float3 e2 = p2 - p0;

    float3 s1 = Cross(ray.dir, e2);
    float divisor = Dot(s1, e1);
    bool hit = true;

    if (divisor == 0.)
        hit = false;
    float invDivisor = 1.f / divisor;

    // Compute first barycentric coordinate
    float3 d = ray.origin - p0;
    float b1 = Dot(d, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        hit = false;

    // Compute second barycentric coordinate
    float3 s2 = Cross(d, e1);
    float b2 = Dot(ray.dir, s2) * invDivisor;
    if (b2 < 0. || b1 + b2 > 1.)
        hit = false;

    // Compute _t_ to intersection point
    float t = Dot(e2, s2) * invDivisor;
    if (t < ray.mint || t > ray.maxt)
        hit = false;

    if (hit) {
        ray.maxt = t;
        ray.hitId = tri.id;
    }
    return hit;
}


bool BVHIntersect(const uniform LinearBVHNode nodes[], 
                  const uniform Triangle tris[], Ray &r) {
    Ray ray = r;
    bool hit = false;
    // Follow ray through BVH nodes to find primitive intersections
    uniform int todoOffset = 0, nodeNum = 0;
    uniform int todo[64];

    while (true) {
        // Check ray against BVH node
        uniform LinearBVHNode node = nodes[nodeNum];
        if (any(BBoxIntersect(node.bounds, ray))) {
            uniform unsigned int nPrimitives = node.nPrimitives;
            if (nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                uniform unsigned int primitivesOffset = node.offset;
                for (uniform unsigned int i = 0; i < nPrimitives; ++i) {
                    if (TriIntersect(tris[primitivesOffset+i], ray))
                        hit = true;
                }
                if (todoOffset == 0) 
                    break;
                nodeNum = todo[--todoOffset];
            }
            else {
                // Put far BVH node on _todo_ stack, advance to near node
                if (r.dirIsNeg[node.splitAxis]) {
                   todo[todoOffset++] = nodeNum + 1;
                   nodeNum = node.offset;
                }
                else {
                   todo[todoOffset++] = node.offset;
                   nodeNum = nodeNum + 1;
                }
            }
        }
        else {
            if (todoOffset == 0)
                break;
            nodeNum = todo[--todoOffset];
        }
    }
    r.maxt = ray.maxt;
    r.hitId = ray.hitId;

    return hit;
}


static void raytrace_tile(uniform int x0, uniform int x1,
                          uniform int y0, uniform int y1, 
                          uniform int width, uniform int height,
                          uniform int baseWidth, uniform int baseHeight,
                          const uniform float raster2camera[4][4], 
                          const uniform float camera2world[4][4],
                          uniform float image[], uniform int id[],
                          const uniform LinearBVHNode nodes[],
                          const uniform Triangle triangles[]) {
    uniform float widthScale = (float)(baseWidth) / (float)(width);
    uniform float heightScale = (float)(baseHeight) / (float)(height);

    foreach_tiled (y = y0 ... y1, x = x0 ... x1) {
        Ray ray;
        generateRay(raster2camera, camera2world, x*widthScale,
                    y*heightScale, ray);
        BVHIntersect(nodes, triangles, ray);

        int offset = y * width + x;
        image[offset] = ray.maxt;
        id[offset] = ray.hitId;
    }
}


export void raytrace_ispc(uniform int width, uniform int height,
                          uniform int baseWidth, uniform int baseHeight,
                          const uniform float raster2camera[4][4], 
                          const uniform float camera2world[4][4],
                          uniform float image[], uniform int id[],
                          const uniform LinearBVHNode nodes[],
                          const uniform Triangle triangles[]) {
    raytrace_tile(0, width, 0, height, width, height, baseWidth, baseHeight,
                  raster2camera, camera2world, image,
                  id, nodes, triangles);
}


task void raytrace_tile_task(uniform int width, uniform int height,
                             uniform int baseWidth, uniform int baseHeight,
                             const uniform float raster2camera[4][4], 
                             const uniform float camera2world[4][4],
                             uniform float image[], uniform int id[],
                             const uniform LinearBVHNode nodes[],
                             const uniform Triangle triangles[]) {
    uniform int dx = 16, dy = 16; // must match dx, dy below
    uniform int xBuckets = (width + (dx-1)) / dx;
    uniform int x0 = (taskIndex % xBuckets) * dx;
    uniform int x1 = min(x0 + dx, width);
    uniform int y0 = (taskIndex / xBuckets) * dy;
    uniform int y1 = min(y0 + dy, height);
                             
    raytrace_tile(x0, x1, y0, y1, width, height, baseWidth, baseHeight, 
                  raster2camera, camera2world, image,
                  id, nodes, triangles);
}


export void raytrace_ispc_tasks(uniform int width, uniform int height,
                                uniform int baseWidth, uniform int baseHeight,
                                const uniform float raster2camera[4][4], 
                                const uniform float camera2world[4][4],
                                uniform float image[], uniform int id[],
                                const uniform LinearBVHNode nodes[],
                                const uniform Triangle triangles[]) {
    uniform int dx = 16, dy = 16;
    uniform int xBuckets = (width + (dx-1)) / dx;
    uniform int yBuckets = (height + (dy-1)) / dy;
    uniform int nTasks = xBuckets * yBuckets;
    launch[nTasks] raytrace_tile_task(width, height, baseWidth, baseHeight, 
                                      raster2camera, camera2world, 
                                      image, id, nodes, triangles);
}

