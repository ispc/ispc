// RUN: %{ispc} %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x4 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s
// RUN: %{ispc} %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x8 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s
// RUN: %{ispc} %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x4 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s
// RUN: %{ispc} %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x8 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: add_d1
// CHECK-COUNT-1: vaddpd
// CHECK-LABEL: add_d2
// CHECK-COUNT-1: vaddpd
// CHECK-NOT: vmovapd
// CHECK-NOT: vmovaps

// CHECK-LABEL: add_f1
// CHECK-COUNT-1: vaddps
// CHECK-LABEL: add_f2
// CHECK-COUNT-1: vaddps
// CHECK-NOT: vmovapd
// CHECK-NOT: vmovlps

typedef struct { uniform double<2> data; } DVec2;

unmasked uniform DVec2 add_d1(uniform DVec2 A, uniform DVec2 B)
{
    uniform double<2> S0, S1, Result;
    *((uniform double<2> *uniform)&S0) = *((uniform double<2> *uniform)&A);
    *((uniform double<2> *uniform)&S1) = *((uniform double<2> *uniform)&B);
    Result = S0 + S1;
    return *((uniform DVec2 *uniform)&Result);
}

unmasked uniform DVec2 add_d2(uniform DVec2 A, uniform DVec2 B)
{
    uniform DVec2 Result;
    foreach(i = 0 ... 2)
    {
        Result.data[i] = A.data[i] + B.data[i];
    }
    return Result;
}

typedef struct { uniform float<3> data; } FVec3;

unmasked uniform FVec3 add_f1(uniform FVec3 A, uniform FVec3 B)
{
    uniform float<3> S0, S1, Result;
    *((uniform float<3> *uniform)&S0) = *((uniform float<3> *uniform)&A);
    *((uniform float<3> *uniform)&S1) = *((uniform float<3> *uniform)&B);
    Result = S0 + S1;
    return *((uniform FVec3 *uniform)&Result);
}

unmasked uniform FVec3 add_f2(uniform FVec3 A, uniform FVec3 B)
{
    uniform FVec3 Result;
    foreach(i = 0 ... 3)
    {
        Result.data[i] = A.data[i] + B.data[i];
    }
    return Result;
}