// -*- mode: c++ -*-
/*
   Copyright (c) 2010-2014, Intel Corporation
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
/*
   Based on Syoyo Fujita's aobench: http://code.google.com/p/aobench
   */

#include "cuda_helpers.cuh"

#define NAO_SAMPLES        8
//#define M_PI 3.1415926535f

#define vec Float3
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

///////////////////////////////////////////////////////////////////////////
// RNG stuff

struct RNGState {
    unsigned int z1, z2, z3, z4;
};

__device__
static inline unsigned int random(RNGState * state)
{
    unsigned int b;

    b  = ((state->z1 << 6) ^ state->z1) >> 13;
    state->z1 = ((state->z1 & 4294967294U) << 18) ^ b;
    b  = ((state->z2 << 2) ^ state->z2) >> 27;
    state->z2 = ((state->z2 & 4294967288U) << 2) ^ b;
    b  = ((state->z3 << 13) ^ state->z3) >> 21;
    state->z3 = ((state->z3 & 4294967280U) << 7) ^ b;
    b  = ((state->z4 << 3) ^ state->z4) >> 12;
    state->z4 = ((state->z4 & 4294967168U) << 13) ^ b;
    return (state->z1 ^ state->z2 ^ state->z3 ^ state->z4);
}


__device__
static inline float frandom(RNGState * state)
{
    unsigned int irand = random(state);
    irand &= (1ul<<23)-1;
    return __int_as_float(0x3F800000 | irand)-1.0f;
}

__device__
static inline void seed_rng(RNGState * state,
                            unsigned int seed) {
    state->z1 = seed;
    state->z2 = seed ^ 0xbeeff00d;
    state->z3 = ((seed & 0xfffful) << 16) | (seed >> 16);
    state->z4 = (((seed & 0xfful) << 24) | ((seed & 0xff00ul)  << 8) |
                 ((seed & 0xff0000ul) >> 8) | (seed & 0xff000000ul) >> 24);
}



struct Isect {
  float      t;
  vec        p;
  vec        n;
  int        hit;
};

struct Sphere {
  vec        center;
  float      radius;
};

struct Plane {
  vec    p;
  vec    n;
};

struct Ray {
  vec org;
  vec dir;
};

__device__
static inline float dot(vec a, vec b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
static inline vec vcross(vec v0, vec v1) {
  vec ret;
  ret.x = v0.y * v1.z - v0.z * v1.y;
  ret.y = v0.z * v1.x - v0.x * v1.z;
  ret.z = v0.x * v1.y - v0.y * v1.x;
  return ret;
}

__device__
static inline void vnormalize(vec &v) {
  float len2 = dot(v, v);
  float invlen = rsqrt(len2);
  v = v*invlen;
}


__device__
static inline void
ray_plane_intersect(Isect &isect,const  Ray &ray, const  Plane &plane) {
  float d = -dot(plane.p, plane.n);
  float v = dot(ray.dir, plane.n);

#if 0
  if (abs(v) < 1.0f-17)
    return;
  else {
    float t = -(dot(ray.org, plane.n) + d) / v;

    if ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org + ray.dir * t;
      isect.n = plane.n;
    }
  }
#else
    if (abs(v) <= 1.0e-17)
      return;
    float t = -(dot(ray.org, plane.n) + d) / v;
    if ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org + ray.dir * t;
      isect.n = plane.n;
    }
#endif
}


__device__
static inline void
ray_sphere_intersect(Isect &isect,const  Ray &ray, const Sphere &sphere) {
  vec rs = ray.org - sphere.center;

  float B = dot(rs, ray.dir);
  float C = dot(rs, rs) - sphere.radius * sphere.radius;
  float D = B * B - C;

#if 0
  if (D > 0.) {
    float t = -B - sqrt(D);

    if ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org +  ray.dir * t;
      isect.n = isect.p - sphere.center;
      vnormalize(isect.n);
    }
  }
#else
    if (D <= 0.0f)
      return;

    float t = -B - sqrt(D);

    if ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org +  ray.dir * t;
      isect.n = isect.p - sphere.center;
      vnormalize(isect.n);
    }
#endif

}


__device__
static inline void
orthoBasis(vec basis[3], vec n) {
  basis[2] = n;
  basis[1].x = 0.0f; basis[1].y = 0.0f; basis[1].z = 0.0f;

  if ((n.x < 0.6f) && (n.x > -0.6f)) {
    basis[1].x = 1.0f;
  } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
    basis[1].y = 1.0f;
  } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
    basis[1].z = 1.0f;
  } else {
    basis[1].x = 1.0f;
  }

  basis[0] = vcross(basis[1], basis[2]);
  vnormalize(basis[0]);

  basis[1] = vcross(basis[2], basis[0]);
  vnormalize(basis[1]);
}


__device__
static inline float
ambient_occlusion(Isect &isect,  const Plane &plane, const  Sphere spheres[3],
    RNGState &rngstate) {
  float eps = 0.0001f;
  vec p; //, n;
  vec basis[3];
  float occlusion = 0.0f;

  p = isect.p + isect.n * eps;

  orthoBasis(basis, isect.n);

  const  int ntheta = NAO_SAMPLES;
  const  int nphi   = NAO_SAMPLES;
  for ( int j = 0; j < ntheta; j++) {
    for ( int i = 0; i < nphi; i++) {
      Ray ray;
      Isect occIsect;

      float theta = sqrt(frandom(&rngstate));
      float phi   = 2.0f * M_PI * frandom(&rngstate);
      float x = cos(phi) * theta;
      float y = sin(phi) * theta;
      float z = sqrtf(1.0f - theta * theta);

      // local . global
      float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

      ray.org = p;
      ray.dir.x = rx;
      ray.dir.y = ry;
      ray.dir.z = rz;

      occIsect.t   = 1.0f+17;
      occIsect.hit = 0;

      for ( int snum = 0; snum < 3; ++snum)
        ray_sphere_intersect(occIsect, ray, spheres[snum]);
      ray_plane_intersect (occIsect, ray, plane);

      if (occIsect.hit) occlusion += 1.0f;
    }
  }

  occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
  return occlusion;
}


/* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
   */
__device__
static inline void ao_tiles(
     int x0,  int x1,
     int y0,  int y1,
     int w,  int h,
     int nsubsamples,
     float image[])
{
  const  Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
  const  Sphere spheres[3] = {
    { { -2.0f, 0.0f, -3.5f }, 0.5f },
    { { -0.5f, 0.0f, -3.0f }, 0.5f },
    { { 1.0f, 0.0f, -2.2f }, 0.5f } };
  RNGState rngstate;

  seed_rng(&rngstate, programIndex + (y0 << (programIndex & 15)));
  float invSamples = 1.f / nsubsamples;
  for ( int y = y0; y < y1; y++)
    for ( int x = programIndex+x0; x < x1; x += programCount)
    {
      const int offset = 3 * (y * w + x);
      float res = 0.0f;

      for ( int u = 0; u < nsubsamples; u++)
        for ( int v = 0; v < nsubsamples; v++)
        {
          float du = (float)u * invSamples, dv = (float)v * invSamples;

          // Figure out x,y pixel in NDC
          float px =  (x + du - (w / 2.0f)) / (w / 2.0f);
          float py = -(y + dv - (h / 2.0f)) / (h / 2.0f);
          float ret = 0.f;
          Ray ray;
          Isect isect;

          ray.org.x = 0.0f;
          ray.org.y = 0.0f;
          ray.org.z = 0.0f;

          // Poor man's perspective projection
          ray.dir.x = px;
          ray.dir.y = py;
          ray.dir.z = -1.0;
          vnormalize(ray.dir);

          isect.t   = 1.0e+17;
          isect.hit = 0;

          for ( int snum = 0; snum < 3; ++snum)
            ray_sphere_intersect(isect, ray, spheres[snum]);
          ray_plane_intersect(isect, ray, plane);

          // Note use of 'coherent' if statement; the set of rays we
          // trace will often all hit or all miss the scene
          if (any(isect.hit)) {
            ret = isect.hit*ambient_occlusion(isect, plane, spheres, rngstate);
            ret *= invSamples * invSamples;
            res += ret;
          }
        }

      if (x < x1)
      {
        image[offset  ] = res;
        image[offset+1] = res;
        image[offset+2] = res;
      }
    }
}



#define TILEX 64
#define TILEY 4

extern "C"
__global__
void ao_task( int width,  int height,
     int nsubsamples,  float image[])
{
  if (taskIndex0 >= taskCount0) return;
  if (taskIndex1 >= taskCount1) return;

  const  int x0 = taskIndex0 * TILEX;
  const  int x1 = min(x0 + TILEX, width);

  const  int y0 = taskIndex1 * TILEY;
  const  int y1 = min(y0 + TILEY, height);
  ao_tiles(x0,x1,y0,y1, width, height, nsubsamples, image);
}

extern "C"
__global__
void ao_ispc_tasks___export(
    int w, int h, int nsubsamples,
    float image[])
{
  const int ntilex = (w+TILEX-1)/TILEX;
  const int ntiley = (h+TILEY-1)/TILEY;
  launch(ntilex,ntiley,1,ao_task)(w,h,nsubsamples,image);
  cudaDeviceSynchronize();
}

extern "C"
__host__ void ao_ispc_tasks(
    int w, int h, int nsubsamples,
    float image[])
{
  ao_ispc_tasks___export<<<1,32>>>(w,h,nsubsamples,image);
  cudaDeviceSynchronize();
}
