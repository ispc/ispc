#include "cuda_helpers.cuh"

#define float3 Float3
struct Float3
{
  float x,y,z;
  __device__ friend Float3 operator+(const Float3 a, const Float3 b)
  {
    Float3 c;
    c.x = a.x+b.x;
    c.y = a.y+b.y;
    c.z = a.z+b.z;
    return c;
  }
  __device__ friend Float3 operator-(const Float3 a, const Float3 b)
  {
    Float3 c;
    c.x = a.x-b.x;
    c.y = a.y-b.y;
    c.z = a.z-b.z;
    return c;
  }
  __device__ friend Float3 operator/(const Float3 a, const Float3 b)
  {
    Float3 c;
    c.x = a.x/b.x;
    c.y = a.y/b.y;
    c.z = a.z/b.z;
    return c;
  }
  __device__ friend Float3 operator/(const float a, const Float3 b)
  {
    Float3 c;
    c.x = a/b.x;
    c.y = a/b.y;
    c.z = a/b.z;
    return c;
  }
  __device__ friend Float3 operator*(const Float3 a, const Float3 b)
  {
    Float3 c;
    c.x = a.x*b.x;
    c.y = a.y*b.y;
    c.z = a.z*b.z;
    return c;
  }
  __device__ friend Float3 operator*(const Float3 a, const float b)
  {
    Float3 c;
    c.x = a.x*b;
    c.y = a.y*b;
    c.z = a.z*b;
    return c;
  }
};

#define int8 char
#define int16 short

struct Ray {
    float3 origin, dir, invDir;
    unsigned int dirIsNeg0, dirIsNeg1, dirIsNeg2;
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

__device__
static inline float3 Cross(const float3 v1, const float3 v2) {
    float v1x = v1.x, v1y = v1.y, v1z = v1.z;
    float v2x = v2.x, v2y = v2.y, v2z = v2.z;
    float3 ret;
    ret.x = (v1y * v2z) - (v1z * v2y);
    ret.y = (v1z * v2x) - (v1x * v2z);
    ret.z = (v1x * v2y) - (v1y * v2x);
    return ret;
}

__device__
static inline float Dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
inline
static void generateRay( const float raster2camera[4][4],
                         const float camera2world[4][4],
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

#if 0
    ray.dirIsNeg[0] = any(ray.invDir.x < 0) ? 1 : 0;
    ray.dirIsNeg[1] = any(ray.invDir.y < 0) ? 1 : 0;
    ray.dirIsNeg[2] = any(ray.invDir.z < 0) ? 1 : 0;
#else
    ray.dirIsNeg0 = any(ray.invDir.x < 0) ? 1 : 0;
    ray.dirIsNeg1 = any(ray.invDir.y < 0) ? 1 : 0;
    ray.dirIsNeg2 = any(ray.invDir.z < 0) ? 1 : 0;
#endif
}

__device__
inline
static bool BBoxIntersect(const  float bounds[2][3],
                          const Ray &ray) {
     float3 bounds0 = { bounds[0][0], bounds[0][1], bounds[0][2] };
     float3 bounds1 = { bounds[1][0], bounds[1][1], bounds[1][2] };
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


__device__
inline
static bool TriIntersect(const  Triangle &tri, Ray &ray) {
     float3 p0 = { tri.p[0][0], tri.p[0][1], tri.p[0][2] };
     float3 p1 = { tri.p[1][0], tri.p[1][1], tri.p[1][2] };
     float3 p2 = { tri.p[2][0], tri.p[2][1], tri.p[2][2] };
     float3 e1 = p1 - p0;
     float3 e2 = p2 - p0;

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

__device__
inline
bool BVHIntersect(const  LinearBVHNode nodes[],
                  const  Triangle tris[], Ray &r,
                   int todo[]) {
    Ray ray = r;
    bool hit = false;
    // Follow ray through BVH nodes to find primitive intersections
     int todoOffset = 0, nodeNum = 0;

    while (true) {
        // Check ray against BVH node
         LinearBVHNode node = nodes[nodeNum];
        if (any(BBoxIntersect(node.bounds, ray))) {
             unsigned int nPrimitives = node.nPrimitives;
            if (nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                 unsigned int primitivesOffset = node.offset;
                for ( unsigned int i = 0; i < nPrimitives; ++i) {
                    if (TriIntersect(tris[primitivesOffset+i], ray))
                        hit = true;
                }
                if (todoOffset == 0)
                    break;
                nodeNum = todo[--todoOffset];
            }
            else {
                // Put far BVH node on _todo_ stack, advance to near node
                int dirIsNeg;
                if (node.splitAxis == 0) dirIsNeg = r.dirIsNeg0;
                if (node.splitAxis == 1) dirIsNeg = r.dirIsNeg1;
                if (node.splitAxis == 2) dirIsNeg = r.dirIsNeg2;
                if (dirIsNeg) {
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

__device__
inline
static void raytrace_tile( int x0,  int x1,
                           int y0,  int y1,
                           int width,  int height,
                           int baseWidth,  int baseHeight,
                          const  float raster2camera[4][4],
                          const  float camera2world[4][4],
                           float image[],  int id[],
                          const  LinearBVHNode nodes[],
                          const  Triangle triangles[]) {
     float widthScale = (float)(baseWidth) / (float)(width);
     float heightScale = (float)(baseHeight) / (float)(height);

#if 0
   int *  todo =  new  int[64];
#define ALLOC
#else
   int todo[64];
#endif

    for (int y = y0 ;y < y1; y++)
      for (int x = x0 + programIndex; x < x1; x += programCount)
        if (x < x1)
        {
          Ray ray;
          generateRay(raster2camera, camera2world, x*widthScale,
              y*heightScale, ray);
          BVHIntersect(nodes, triangles, ray, todo);

          int offset = y * width + x;
          image[offset] = ray.maxt;
          id[offset] = ray.hitId;
        }

#ifdef ALLOC
  delete todo;
#endif
}



__global__
void raytrace_tile_task( int width,  int height,
                              int baseWidth,  int baseHeight,
                             const  float raster2camera[4][4],
                             const  float camera2world[4][4],
                              float image[],  int id[],
                             const  LinearBVHNode nodes[],
                             const  Triangle triangles[]) {
     int dx = 64, dy = 8; // must match dx, dy below
     int xBuckets = (width + (dx-1)) / dx;
     int x0 = (taskIndex % xBuckets) * dx;
     int x1 = min(x0 + dx, width);
     int y0 = (taskIndex / xBuckets) * dy;
     int y1 = min(y0 + dy, height);

    raytrace_tile(x0, x1, y0, y1, width, height, baseWidth, baseHeight,
                  raster2camera, camera2world, image,
                  id, nodes, triangles);
}


extern "C" __global__ void raytrace_ispc_tasks___export( int width,  int height,
                                 int baseWidth,  int baseHeight,
                                const  float raster2camera[4][4],
                                const  float camera2world[4][4],
                                 float image[],  int id[],
                                const  LinearBVHNode nodes[],
                                const  Triangle triangles[]) {
     int dx = 64, dy = 8;
     int xBuckets = (width + (dx-1)) / dx;
     int yBuckets = (height + (dy-1)) / dy;
     int nTasks = xBuckets * yBuckets;
     launch(nTasks,1,1,raytrace_tile_task)
       (width, height, baseWidth, baseHeight,
        raster2camera, camera2world,
        image, id, nodes, triangles);
     cudaDeviceSynchronize();
}



extern "C" __host__ void raytrace_ispc_tasks( int width,  int height,
    int baseWidth,  int baseHeight,
    const  float raster2camera[4][4],
    const  float camera2world[4][4],
    float image[],  int id[],
    const  LinearBVHNode nodes[],
    const  Triangle triangles[]) {
  raytrace_ispc_tasks___export<<<1,32>>>( width,  height,
      baseWidth,  baseHeight,
      raster2camera,
      camera2world,
      image,  id,
      nodes,
      triangles);
  cudaDeviceSynchronize();
}
