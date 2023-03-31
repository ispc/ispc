// -*- mode: c++ -*-
/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/
/*
  Based on Syoyo Fujita's aobench: http://code.google.com/p/aobench
*/

#define NAO_SAMPLES		8
#define M_PI 3.1415926535f

typedef float<3> vec;

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


static inline void
ray_plane_intersect(Isect &isect, Ray &ray, Plane &plane) {
    float d = -dot(plane.p, plane.n);
    float v = dot(ray.dir, plane.n);

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
}


static inline void
ray_sphere_intersect(Isect &isect, Ray &ray, Sphere &sphere) {
    vec rs = ray.org - sphere.center;

    float B = dot(rs, ray.dir);
    float C = dot(rs, rs) - sphere.radius * sphere.radius;
    float D = B * B - C;

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
}


static inline void
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


static inline float
ambient_occlusion(Isect &isect, Plane &plane, Sphere spheres[3],
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


/* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
 */
static void ao_scanlines(uniform int y0, uniform int y1, uniform int x0, uniform int x1, uniform int w,
                         uniform int h,  uniform int nsubsamples,
                         uniform float image[]) {
    static Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
    static Sphere spheres[3] = {
        { { -2.0f, 0.0f, -3.5f }, 0.5f },
        { { -0.5f, 0.0f, -3.0f }, 0.5f },
        { { 1.0f, 0.0f, -2.2f }, 0.5f } };
    RNGState rngstate;

    seed_rng(&rngstate, programIndex + (y0 << (programIndex & 15)));

    // Compute the mapping between the 'programCount'-wide program
    // instances running in parallel and samples in the image.
    //
    // For now, we'll always take four samples per pixel, so start by
    // initializing du and dv with offsets into subpixel samples.  We'll
    // take care of further updating du and dv for the case where we're
    // doing more than 4 program instances in parallel shortly.
    uniform float uSteps[4] = { 0, 1, 0, 1 };
    uniform float vSteps[4] = { 0, 0, 1, 1 };
    float du = uSteps[programIndex % 4] / nsubsamples;
    float dv = vSteps[programIndex % 4] / nsubsamples;

    // Now handle the case where we are able to do more than one pixel's
    // worth of work at once.  nx records the number of pixels in the x
    // direction we do per iteration and ny the number in y.
    uniform int nx = 1, ny = 1;

    // FIXME: We actually need ny to be 1 regardless of the decomposition,
    // since the task decomposition is one scanline high.

    if (programCount == 8) {
        // Do two pixels at once in the x direction
        nx = 2;
        if (programIndex >= 4)
            // And shift the offsets for the second pixel's worth of work
            ++du;
    }
    else if (programCount == 16) {
        nx = 4;
        ny = 1;
        if (programIndex >= 4 && programIndex < 8)
            ++du;
        if (programIndex >= 8 && programIndex < 12)
            du += 2;
        if (programIndex >= 12)
            du += 3;
    }

    // Now loop over all of the pixels, stepping in x and y as calculated
    // above.  (Assumes that ny divides y and nx divides x...)
    uniform int y = y0;
    for (uniform int x = x0; x < x1; x += nx)  {
        // Figure out x,y pixel in NDC
        float px =  (x + du - (w / 2.0f)) / (w / 2.0f);
        float py = -(y + dv - (h / 2.0f)) / (h / 2.0f);

        // Scale NDC based on width/height ratio, supporting non-square image output
        px *= (float)w / (float)h;

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
        cif (isect.hit)
            ret = ambient_occlusion(isect, plane, spheres, rngstate);

        // This is a little grungy; we have results for
        // programCount-worth of values.  Because we're doing 2x2
        // subsamples, we need to peel them off in groups of four,
        // average the four values for each pixel, and update the
        // output image.
        //
        // Store the varying value to a uniform array of the same size.
        // See the discussion about communication among program
        // instances in the ispc user's manual for more discussion on
        // this idiom.
        uniform float retArray[programCount];
        retArray[programIndex] = ret;

        // offset to the first pixel in the image
        uniform int offset = 3 * (y * w + x);
        for (uniform int p = 0; p < programCount; p += 4, offset += 3) {
            // Get the four sample values for this pixel
            uniform float sumret = retArray[p] + retArray[p+1] + retArray[p+2] +
                retArray[p+3];

            // Normalize by number of samples taken
            sumret /= nsubsamples * nsubsamples;

            // Store result in the image
            image[offset+0] = sumret;
            image[offset+1] = sumret;
            image[offset+2] = sumret;
        }
    }
}

struct Parameters {
    int width;
    int height;
    int nsubsamples;
    float *image;
};

task void ao_ispc(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;

    uniform int workPerThread = (p->width*p->height)/taskCount;
    uniform int workItemsPerLine = p->width / workPerThread;
    // this assumes thread does less or equal work to a single line
    uniform int start = (taskIndex*workPerThread)/p->width;
    uniform int end = start + 1;
    uniform int x_start = (taskIndex*workPerThread) % p->width;
    uniform int x_end = x_start + workPerThread;

    ao_scanlines(start, end, x_start, x_end, p->width, p->height, p->nsubsamples, p->image);
}

#include "ispcrt.isph"
DEFINE_CPU_ENTRY_POINT(ao_ispc)
