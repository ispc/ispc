// This test checks that --opt=disable-zmm is supported for generic targets

// RUN: %{ispc} -O2 --target=generic-i1x16 --opt=disable-zmm --cpu=spr --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s
// RUN: %{ispc} -O2 --target=generic-i1x32 --opt=disable-zmm --cpu=skx --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s
// RUN: %{ispc} -O2 --target=generic-i1x64 --opt=disable-zmm --cpu=icl --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: ymm
// CHECK-NOT: zmm

export void Max(uniform float Result[], const uniform float Source1[], const uniform float Source2[], const uniform int Iterations)
{
    foreach(i = 0 ... Iterations)
    {
        Result[i] = max(Source1[i], Source2[i]);
    }
}
