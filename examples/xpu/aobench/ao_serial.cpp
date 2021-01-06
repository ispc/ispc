// -*- mode: c++ -*-
/*
  Copyright (c) 2010-2020, Intel Corporation
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

#include <math.h>
#include <stdlib.h>

#ifdef _MSC_VER
static long long drand48_x = 0x1234ABCD330E;

static inline void srand48(int x) { drand48_x = x ^ (x << 16); }

static inline double drand48() {
    drand48_x = drand48_x * 0x5DEECE66D + 0xB;
    return (drand48_x & 0xFFFFFFFFFFFF) * (1.0 / 281474976710656.0);
}
#endif // _MSC_VER

#ifdef _MSC_VER
__declspec(align(16))
#endif
    struct vec {
    vec() { x = y = z = pad = 0.; }
    vec(float xx, float yy, float zz) {
        x = xx;
        y = yy;
        z = zz;
        pad = 0;
    }

    vec operator*(float f) const { return vec(x * f, y * f, z * f); }
    vec operator+(const vec &f2) const { return vec(x + f2.x, y + f2.y, z + f2.z); }
    vec operator-(const vec &f2) const { return vec(x - f2.x, y - f2.y, z - f2.z); }
    vec operator*(const vec &f2) const { return vec(x * f2.x, y * f2.y, z * f2.z); }
    float x, y, z;
    float pad;
}
#ifndef _MSC_VER
__attribute__((aligned(16)))
#endif
;
inline vec operator*(float f, const vec &v) { return vec(f * v.x, f * v.y, f * v.z); }

#define NAO_SAMPLES 8

#ifdef M_PI
#undef M_PI
#endif
#define M_PI 3.1415926535f

struct Isect {
    float t;
    vec p;
    vec n;
    int hit;
};

struct Sphere {
    vec center;
    float radius;
};

struct Plane {
    vec p;
    vec n;
};

struct Ray {
    vec org;
    vec dir;
};

static inline float dot(const vec &a, const vec &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline vec vcross(const vec &v0, const vec &v1) {
    vec ret;
    ret.x = v0.y * v1.z - v0.z * v1.y;
    ret.y = v0.z * v1.x - v0.x * v1.z;
    ret.z = v0.x * v1.y - v0.y * v1.x;
    return ret;
}

static inline void vnormalize(vec &v) {
    float len2 = dot(v, v);
    float invlen = 1.f / sqrtf(len2);
    v = v * invlen;
}

static inline void ray_plane_intersect(Isect &isect, Ray &ray, Plane &plane) {
    float d = -dot(plane.p, plane.n);
    float v = dot(ray.dir, plane.n);

    if (fabsf(v) < 1.0e-17f)
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
}

static inline void ray_sphere_intersect(Isect &isect, Ray &ray, Sphere &sphere) {
    vec rs = ray.org - sphere.center;

    float B = dot(rs, ray.dir);
    float C = dot(rs, rs) - sphere.radius * sphere.radius;
    float D = B * B - C;

    if (D > 0.) {
        float t = -B - sqrtf(D);

        if ((t > 0.0) && (t < isect.t)) {
            isect.t = t;
            isect.hit = 1;
            isect.p = ray.org + t * ray.dir;
            isect.n = isect.p - sphere.center;
            vnormalize(isect.n);
        }
    }
}

static inline void orthoBasis(vec basis[3], const vec &n) {
    basis[2] = n;
    basis[1].x = 0.0;
    basis[1].y = 0.0;
    basis[1].z = 0.0;

    if ((n.x < 0.6f) && (n.x > -0.6f)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    basis[0] = vcross(basis[1], basis[2]);
    vnormalize(basis[0]);

    basis[1] = vcross(basis[2], basis[0]);
    vnormalize(basis[1]);
}

static float ambient_occlusion(Isect &isect, Plane &plane, Sphere spheres[3]) {
    float eps = 0.0001f;
    vec p, n;
    vec basis[3];
    float occlusion = 0.0;

    p = isect.p + eps * isect.n;

    orthoBasis(basis, isect.n);

    static const int ntheta = NAO_SAMPLES;
    static const int nphi = NAO_SAMPLES;
    for (int j = 0; j < ntheta; j++) {
        for (int i = 0; i < nphi; i++) {
            Ray ray;
            Isect occIsect;

            float theta = sqrtf(drand48());
            float phi = 2.0f * M_PI * drand48();
            float x = cosf(phi) * theta;
            float y = sinf(phi) * theta;
            float z = sqrtf(1.0f - theta * theta);

            // local . global
            float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            ray.org = p;
            ray.dir.x = rx;
            ray.dir.y = ry;
            ray.dir.z = rz;

            occIsect.t = 1.0e+17f;
            occIsect.hit = 0;

            for (int snum = 0; snum < 3; ++snum)
                ray_sphere_intersect(occIsect, ray, spheres[snum]);
            ray_plane_intersect(occIsect, ray, plane);

            if (occIsect.hit)
                occlusion += 1.f;
        }
    }

    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
    return occlusion;
}

/* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
 */
static void ao_scanlines(int y0, int y1, int w, int h, int nsubsamples, float image[]) {
    static Plane plane = {vec(0.0f, -0.5f, 0.0f), vec(0.f, 1.f, 0.f)};
    static Sphere spheres[3] = {
        {vec(-2.0f, 0.0f, -3.5f), 0.5f}, {vec(-0.5f, 0.0f, -3.0f), 0.5f}, {vec(1.0f, 0.0f, -2.2f), 0.5f}};

    srand48(y0);

    for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < w; ++x) {
            int offset = 3 * (y * w + x);
            for (int u = 0; u < nsubsamples; ++u) {
                for (int v = 0; v < nsubsamples; ++v) {
                    float px = (x + (u / (float)nsubsamples) - (w / 2.0f)) / (w / 2.0f);
                    float py = -(y + (v / (float)nsubsamples) - (h / 2.0f)) / (h / 2.0f);

                    // Scale NDC based on width/height ratio, supporting non-square image output
                    px *= (float)w / (float)h;

                    float ret = 0.f;
                    Ray ray;
                    Isect isect;

                    ray.org = vec(0.f, 0.f, 0.f);

                    ray.dir.x = px;
                    ray.dir.y = py;
                    ray.dir.z = -1.0f;
                    vnormalize(ray.dir);

                    isect.t = 1.0e+17f;
                    isect.hit = 0;

                    for (int snum = 0; snum < 3; ++snum)
                        ray_sphere_intersect(isect, ray, spheres[snum]);
                    ray_plane_intersect(isect, ray, plane);

                    if (isect.hit)
                        ret = ambient_occlusion(isect, plane, spheres);

                    // Update image for AO for this ray
                    image[offset + 0] += ret;
                    image[offset + 1] += ret;
                    image[offset + 2] += ret;
                }
            }
            // Normalize image pixels by number of samples taken per pixel
            image[offset + 0] /= nsubsamples * nsubsamples;
            image[offset + 1] /= nsubsamples * nsubsamples;
            image[offset + 2] /= nsubsamples * nsubsamples;
        }
    }
}

void ao_serial(int w, int h, int nsubsamples, float image[]) { ao_scanlines(0, h, w, h, nsubsamples, image); }
