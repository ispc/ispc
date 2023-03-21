// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#define MAX_INSTANCE_LEVEL_COUNT 1
#define INVALID_GEOMETRY_ID -1

template <typename T> using uniform = sycl::ext::oneapi::experimental::uniform<T>;

struct alignas(16) vec3f {
    float x = 0;
    float y = 0;
    float z = 0;
};

vec3f operator*(const vec3f &a, const vec3f &b) { return vec3f{a.x * b.x, a.y * b.y, a.z * b.z}; }

vec3f operator-(const vec3f &a, const vec3f &b) { return vec3f{a.x - b.x, a.y - b.y, a.z - b.z}; }

vec3f operator+(const vec3f &a, const vec3f &b) { return vec3f{a.x + b.x, a.y + b.y, a.z + b.z}; }

vec3f operator*(const vec3f &a, const float s) { return vec3f{a.x * s, a.y * s, a.z * s}; }

vec3f operator*(const float s, const vec3f &a) { return a * s; }

/* Ray structure */
struct Ray {
    float org_x; // x coordinate of ray origin
    float org_y; // y coordinate of ray origin
    float org_z; // z coordinate of ray origin
    float tnear; // start of ray segment

    float dir_x; // x coordinate of ray direction
    float dir_y; // y coordinate of ray direction
    float dir_z; // z coordinate of ray direction
    float time;  // time of this ray for motion blur

    float tfar;        // end of ray segment (set to hit distance)
    unsigned int mask; // ray mask
    unsigned int id;   // ray ID
};

/* Hit structure */
struct Hit {
    float Ng_x; // x coordinate of geometry normal
    float Ng_y; // y coordinate of geometry normal
    float Ng_z; // z coordinate of geometry normal

    float u; // barycentric u coordinate of hit
    float v; // barycentric v coordinate of hit

    unsigned int primID;                           // primitive ID
    unsigned int geomID;                           // geometry ID
    unsigned int instID[MAX_INSTANCE_LEVEL_COUNT]; // instance ID
};

/* Combined ray/hit structure */
struct RayHit {
    Ray ray;
    Hit hit;
};

// SIMD-width specific versions of the ray
/* Ray structure */
template <size_t VL> struct RayV {
    float org_x[VL]; // x coordinate of ray origin
    float org_y[VL]; // y coordinate of ray origin
    float org_z[VL]; // z coordinate of ray origin
    float tnear[VL]; // start of ray segment

    float dir_x[VL]; // x coordinate of ray direction
    float dir_y[VL]; // y coordinate of ray direction
    float dir_z[VL]; // z coordinate of ray direction
    float time[VL];  // time of this ray for motion blur

    float tfar[VL];        // end of ray segment (set to hit distance)
    unsigned int mask[VL]; // ray mask
    unsigned int id[VL];   // ray ID
};

/* Hit structure */
template <size_t VL> struct HitV {
    float Ng_x[VL]; // x coordinate of geometry normal
    float Ng_y[VL]; // y coordinate of geometry normal
    float Ng_z[VL]; // z coordinate of geometry normal

    float u[VL]; // barycentric u coordinate of hit
    float v[VL]; // barycentric v coordinate of hit

    unsigned int primID[VL];                           // primitive ID
    unsigned int geomID[VL];                           // geometry ID
    unsigned int instID[MAX_INSTANCE_LEVEL_COUNT][VL]; // instance ID
};

/* Combined ray/hit structure */
template <size_t VL> struct RayHitV {
    RayV<VL> ray;
    HitV<VL> hit;
};

using Ray16 = RayV<16>;
using Hit16 = HitV<16>;
using RayHit16 = RayHitV<16>;

struct Sphere {
    vec3f center;
    float radius;
};

struct Plane {
    vec3f p, n;
};

struct Scene {
    Plane *planes;
    Sphere *spheres;

    unsigned int n_planes;
    unsigned int n_spheres;
};

inline float dot(vec3f a, vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3f cross(vec3f a, vec3f b) {
    vec3f c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
    return c;
}
inline vec3f normalized(vec3f a) {
    float inv_len = 1.f / sycl::sqrt(dot(a, a));
    return a * inv_len;
}

inline bool sphere_intersect(const Sphere &sphere, RayHit &rayhit) {
    const vec3f ray_org = {rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z};
    const vec3f ray_dir = {rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z};
    vec3f rs = ray_org - sphere.center;

    const float b = dot(rs, ray_dir);
    const float c = dot(rs, rs) - sphere.radius * sphere.radius;
    const float discrim = b * b - c;
    if (discrim > 0.f) {
        // Note: we will never hit the backface of the sphere
        const float t = -b - sycl::sqrt(discrim);
        if (t > rayhit.ray.tnear && t < rayhit.ray.tfar) {
            vec3f hitp = ray_org + ray_dir * t;
            vec3f n = normalized(hitp - sphere.center);

            rayhit.ray.tfar = t;
            rayhit.hit.primID = 0;
            rayhit.hit.Ng_x = n.x;
            rayhit.hit.Ng_y = n.y;
            rayhit.hit.Ng_z = n.z;
            rayhit.hit.u = 0;
            rayhit.hit.v = 0;
            return true;
        }
    }
    return false;
}

inline bool plane_intersect(const Plane &plane, RayHit &rayhit) {
    const vec3f ray_org = {rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z};
    const vec3f ray_dir = {rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z};

    const float d = -dot(plane.p, plane.n);
    const float v = dot(ray_dir, plane.n);
    if (sycl::abs(v) < 1e-6f) {
        return false;
    }

    const float t = -(dot(ray_org, plane.n) + d) / v;
    if (t > rayhit.ray.tnear && t < rayhit.ray.tfar) {
        rayhit.ray.tfar = t;
        rayhit.hit.primID = 0;
        rayhit.hit.Ng_x = plane.n.x;
        rayhit.hit.Ng_y = plane.n.y;
        rayhit.hit.Ng_z = plane.n.z;
        rayhit.hit.u = 0;
        rayhit.hit.v = 0;
        return true;
    }
    return false;
}

inline bool sphere_occluded(const Sphere &sphere, Ray &ray) {
    const vec3f ray_org = {ray.org_x, ray.org_y, ray.org_z};
    const vec3f ray_dir = {ray.dir_x, ray.dir_y, ray.dir_z};
    vec3f rs = ray_org - sphere.center;

    const float b = dot(rs, ray_dir);
    const float c = dot(rs, rs) - sphere.radius * sphere.radius;
    const float discrim = b * b - c;
    if (discrim > 0.f) {
        // Note: we will never hit the backface of the sphere because
        // of how the aobench scene is set up
        const float t = -b - sycl::sqrt(discrim);
        if (t > ray.tnear && t < ray.tfar) {
            vec3f hitp = ray_org + ray_dir * t;
            vec3f n = normalized(hitp - sphere.center);

            ray.tfar = t;
            return true;
        }
    }
    return false;
}

inline bool plane_occluded(const Plane &plane, Ray &ray) {
    const vec3f ray_org = {ray.org_x, ray.org_y, ray.org_z};
    const vec3f ray_dir = {ray.dir_x, ray.dir_y, ray.dir_z};

    const float d = -dot(plane.p, plane.n);
    const float v = dot(ray_dir, plane.n);
    if (sycl::abs(v) < 1e-6f) {
        return false;
    }

    const float t = -(dot(ray_org, plane.n) + d) / v;
    if (t > ray.tnear && t < ray.tfar) {
        ray.tfar = t;
        return true;
    }
    return false;
}

extern "C" SYCL_EXTERNAL void __regcall intersect16(const int valid, Scene *scene, RayHit16 *rayhit16) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    if (!valid) {
        return;
    }

    // ISPC and SYCL has different data layout.
    // struct S {
    //   int a;
    //   int b;
    //  }
    // ISPC has SoA layout:
    // S: [a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b]
    // SYCL has AoS layout:
    // S: [a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b]
    // So below we're doing data layout transformation.

    // Load this thread's ray to simplify the code a bit.
    const uint32_t tid = sg.get_local_id();
    RayHit rayhit;
    rayhit.ray.org_x = rayhit16->ray.org_x[tid];
    rayhit.ray.org_y = rayhit16->ray.org_y[tid];
    rayhit.ray.org_z = rayhit16->ray.org_z[tid];

    rayhit.ray.tnear = rayhit16->ray.tnear[tid];

    rayhit.ray.dir_x = rayhit16->ray.dir_x[tid];
    rayhit.ray.dir_y = rayhit16->ray.dir_y[tid];
    rayhit.ray.dir_z = rayhit16->ray.dir_z[tid];

    rayhit.ray.time = rayhit16->ray.time[tid];

    rayhit.ray.tfar = rayhit16->ray.tfar[tid];

    rayhit.ray.mask = rayhit16->ray.mask[tid];
    rayhit.ray.id = rayhit16->ray.id[tid];

    rayhit.hit.primID = INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = INVALID_GEOMETRY_ID;

    // Each object in the "scene" is a different geometry within one instance,
    // so here make up the geom id values.
    unsigned int geom_id = 0;
    for (unsigned int i = 0; i < scene->n_planes; ++i, ++geom_id) {
        if (plane_intersect(scene->planes[i], rayhit)) {
            rayhit.hit.geomID = geom_id;
            rayhit.hit.instID[0] = 0;
        }
    }
    for (unsigned int i = 0; i < scene->n_spheres; ++i, ++geom_id) {
        if (sphere_intersect(scene->spheres[i], rayhit)) {
            rayhit.hit.geomID = geom_id;
            rayhit.hit.instID[0] = 0;
        }
    }

    // Copy back hit results in the right data layout
    rayhit16->hit.Ng_x[tid] = rayhit.hit.Ng_x;
    rayhit16->hit.Ng_y[tid] = rayhit.hit.Ng_y;
    rayhit16->hit.Ng_z[tid] = rayhit.hit.Ng_z;
    rayhit16->hit.u[tid] = rayhit.hit.u;
    rayhit16->hit.v[tid] = rayhit.hit.v;

    rayhit16->hit.primID[tid] = rayhit.hit.primID;
    rayhit16->hit.geomID[tid] = rayhit.hit.geomID;
    rayhit16->hit.instID[0][tid] = rayhit.hit.instID[0];

    rayhit16->ray.tfar[tid] = rayhit.ray.tfar;
}

extern "C" SYCL_EXTERNAL void __regcall occluded16(const int valid, Scene *scene, Ray16 *ray16) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    if (!valid) {
        return;
    }

    // ISPC and SYCL has different data layout.
    // struct S {
    //   int a;
    //   int b;
    //  }
    // ISPC has SoA layout:
    // S: [a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b]
    // SYCL has AoS layout:
    // S: [a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b]
    // So below we're doing data layout transformation.

    // Load this thread's ray to simplify the code a bit.
    const uint32_t tid = sg.get_local_id();
    Ray ray;
    ray.org_x = ray16->org_x[tid];
    ray.org_y = ray16->org_y[tid];
    ray.org_z = ray16->org_z[tid];

    ray.tnear = ray16->tnear[tid];

    ray.dir_x = ray16->dir_x[tid];
    ray.dir_y = ray16->dir_y[tid];
    ray.dir_z = ray16->dir_z[tid];

    ray.time = ray16->time[tid];

    ray.tfar = ray16->tfar[tid];

    ray.mask = ray16->mask[tid];
    ray.id = ray16->id[tid];

    // Each object in the "scene" is a different geometry within one instance,
    // so here make up the geom id values.
    unsigned int geom_id = 0;
    for (unsigned int i = 0; i < scene->n_planes; ++i, ++geom_id) {
        if (plane_occluded(scene->planes[i], ray)) {
            // occluded will set tfar to -inf when a hit is found
            ray.tfar = -1e20f;
            ray16->tfar[tid] = -1e20f;
            return;
        }
    }
    for (unsigned int i = 0; i < scene->n_spheres; ++i, ++geom_id) {
        if (sphere_occluded(scene->spheres[i], ray)) {
            // occluded will set tfar to -inf when a hit is found
            ray.tfar = -1e20f;
            ray16->tfar[tid] = -1e20f;
            return;
        }
    }
}
