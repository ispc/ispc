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

#define NAO_SAMPLES        8
#define M_PI 3.1415926535f

typedef float<3> vec;

#if 1
#define __inline inline
#else
#define __inline
#endif

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

static inline float dot(vec a, vec b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline vec vcross(vec v0, vec v1) {
    vec ret;
    ret.x = v0.y * v1.z - v0.z * v1.y;
    ret.y = v0.z * v1.x - v0.x * v1.z;
    ret.z = v0.x * v1.y - v0.y * v1.x;
    return ret;
}

static inline void vnormalize(vec &v) {
    float len2 = dot(v, v);
    float invlen = rsqrt(len2);
    v *= invlen;
}


__inline
static void
ray_plane_intersect(Isect &isect, Ray &ray, const Plane &plane) {
    float d = -dot(plane.p, plane.n);
    float v = dot(ray.dir, plane.n);

#if 0
    cif (abs(v) < 1.0e-17)
        return;
    else {
        float t = -(dot(ray.org, plane.n) + d) / v;

        cif ((t > 0.0) && (t < isect.t)) {
            isect.t = t;
            isect.hit = 1;
            isect.p = ray.org + ray.dir * t;
            isect.n = plane.n;
        }
    }
#else
    cif (abs(v) <= 1.0e-17)
      return;
    float t = -(dot(ray.org, plane.n) + d) / v;
    cif ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org + ray.dir * t;
      isect.n = plane.n;
    }
#endif
}


static inline void
ray_sphere_intersect(Isect &isect, Ray &ray, const Sphere &sphere) {
    vec rs = ray.org - sphere.center;

    float B = dot(rs, ray.dir);
    float C = dot(rs, rs) - sphere.radius * sphere.radius;
    float D = B * B - C;

#if 0
    cif (D > 0.) {
        float t = -B - sqrt(D);

        cif ((t > 0.0) && (t < isect.t)) {
            isect.t = t;
            isect.hit = 1;
            isect.p = ray.org + t * ray.dir;
            isect.n = isect.p - sphere.center;
            vnormalize(isect.n);
        }
    }
#else
    cif (D <=0.0f)
      return;

    float t = -B - sqrt(D);
    cif ((t > 0.0) && (t < isect.t)) {
      isect.t = t;
      isect.hit = 1;
      isect.p = ray.org + t * ray.dir;
      isect.n = isect.p - sphere.center;
      vnormalize(isect.n);
    }
#endif
}


__inline
static void
orthoBasis(vec basis[3], vec n) {
    basis[2] = n;
    basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    basis[0] = vcross(basis[1], basis[2]);
    vnormalize(basis[0]);

    basis[1] = vcross(basis[2], basis[0]);
    vnormalize(basis[1]);
}


__inline
static float
ambient_occlusion(Isect &isect, const Plane &plane, const Sphere spheres[3],
                  RNGState &rngstate) {
    float eps = 0.0001f;
    vec p, n;
    vec basis[3];
    float occlusion = 0.0;

    p = isect.p + eps * isect.n;

    orthoBasis(basis, isect.n);

    static const uniform int ntheta = NAO_SAMPLES;
    static const uniform int nphi   = NAO_SAMPLES;
    for (uniform int j = 0; j < ntheta; j++) {
        for (uniform int i = 0; i < nphi; i++) {
            Ray ray;
            Isect occIsect;

            float theta = sqrt(frandom(&rngstate));
            float phi   = 2.0f * M_PI * frandom(&rngstate);
            float x = cos(phi) * theta;
            float y = sin(phi) * theta;
            float z = sqrt(1.0 - theta * theta);

            // local . global
            float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            ray.org = p;
            ray.dir.x = rx;
            ray.dir.y = ry;
            ray.dir.z = rz;

            occIsect.t   = 1.0e+17;
            occIsect.hit = 0;

            for (uniform int snum = 0; snum < 3; ++snum)
                ray_sphere_intersect(occIsect, ray, spheres[snum]);
            ray_plane_intersect (occIsect, ray, plane);

            if (occIsect.hit) occlusion += 1.0;
        }
    }

    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
    return occlusion;
}

static  inline void ao_tiles(
    uniform int x0, uniform int x1,
    uniform int y0, uniform int y1,
    uniform int w, uniform int h,
    uniform int nsubsamples,
    uniform float image[])
{
  const Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
  const Sphere spheres[3] = {
    { { -2.0f, 0.0f, -3.5f }, 0.5f },
    { { -0.5f, 0.0f, -3.0f }, 0.5f },
    { { 1.0f, 0.0f, -2.2f }, 0.5f } };
  RNGState rngstate;

  seed_rng(&rngstate, programIndex + (y0 << (programIndex & 15)));
  float invSamples = 1.f / nsubsamples;
  foreach_tiled (y = y0 ... y1, x = x0 ... x1)
  {
    const int offset = 3 * (y * w + x);
    float res = 0.0f;

    for (uniform int u = 0; u < nsubsamples; u++)
      for (uniform int v = 0; v < nsubsamples; v++)
      {
        float du = (float)u * invSamples, dv = (float)v * invSamples;

        // Figure out x,y pixel in NDC
        float px =  (x + du - (w / 2.0f)) / (w / 2.0f);
        float py = -(y + dv - (h / 2.0f)) / (h / 2.0f);
        float ret = 0.f;
        Ray ray;
        Isect isect;

        ray.org = 0.f;

        // Poor man's perspective projection
        ray.dir.x = px;
        ray.dir.y = py;
        ray.dir.z = -1.0;
        vnormalize(ray.dir);

        isect.t   = 1.0e+17;
        isect.hit = 0;

        for (uniform int snum = 0; snum < 3; ++snum)
          ray_sphere_intersect(isect, ray, spheres[snum]);
        ray_plane_intersect(isect, ray, plane);

        // Note use of 'coherent' if statement; the set of rays we
        // trace will often all hit or all miss the scene
#if 0
        cif (isect.hit) {
          ret = ambient_occlusion(isect, plane, spheres, rngstate);
          ret *= invSamples * invSamples;
          res += ret;
        }
#else
         if(any(isect.hit))
         {
          ret = isect.hit*ambient_occlusion(isect, plane, spheres, rngstate);
          ret *= invSamples * invSamples;
          res += ret;
         }
#endif
      }

      image[offset  ] = res;
      image[offset+1] = res;
      image[offset+2] = res;
  }
}

#define TILEX max(64,programCount*2)
#define TILEY 4

export void ao_ispc(uniform int w, uniform int h, uniform int nsubsamples,
                    uniform float image[]) {
  const uniform int x0 = 0;
  const uniform int x1 = w;
  const uniform int y0 = 0;
  const uniform int y1 = h;
  ao_tiles(x0,x1,y0,y1, w, h, nsubsamples, image);
}

void task ao_task(uniform int width, uniform int height,
    uniform int nsubsamples, uniform float image[])
{
  if (taskIndex0 >= taskCount0) return;
  if (taskIndex1 >= taskCount1) return;

  const uniform int x0 = taskIndex0 * TILEX;
  const uniform int x1 = min(x0 + TILEX, width);

  const uniform int y0 = taskIndex1 * TILEY;
  const uniform int y1 = min(y0 + TILEY, height);
  ao_tiles(x0,x1,y0,y1, width, height, nsubsamples, image);
}


export void ao_ispc_tasks(uniform int w, uniform int h, uniform int nsubsamples,
    uniform float image[])
{
  const uniform int ntilex = (w+TILEX-1)/TILEX;
  const uniform int ntiley = (h+TILEY-1)/TILEY;
  launch[ntilex,ntiley] ao_task(w, h, nsubsamples, image);
  sync;
}
