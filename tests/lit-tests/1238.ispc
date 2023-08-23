// RUN: %{ispc} --arch=x86-64 --pic -O3 -DINLINE=inline --target=sse4-i32x8,avx1-i32x8,avx2-i32x8 --nostdlib --nowrap -o %t.ll --emit-llvm-text %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: FATAL ERROR:
struct Point3s {float x, y, z;};
typedef struct Point3s Point3;

#define NOISE_PERM_SIZE 8

static uniform int NoisePerm[2 * NOISE_PERM_SIZE] = {
    151, 160, 137, 91, 90, 15, 131, 13,
    201, 95, 96, 53, 194, 233, 7, 225,
};

inline float Grad(int x, int y, int z) {
    int h = NoisePerm[NoisePerm[NoisePerm[x]+y]+z];
    h &= 15;
    float u = h<8 || h==12 || h==13 ? 1. : 0.;
    return u;
}

export
void noise_ispc (
        uniform Point3  pnt[],
        uniform float output[])
{
    Point3 xseed;
    xseed.x = pnt[programIndex].x;
    xseed.y = pnt[programIndex].y;
    xseed.z = pnt[programIndex].z;
    output[programIndex] = Grad (xseed.x, xseed.y,xseed.z);
    return;
}
