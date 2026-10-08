// With a constant Morton mask, the uniform software pdep loop fully unrolls on
// fallback targets, and BMI2 targets keep the native instruction.
// RUN: %{ispc} %s --target=sse4.2-i32x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,SW --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=avx1-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,SW --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE

// REQUIRES: X86_ENABLED

// CHECK-LABEL: define {{.*}} @morton32___
// SW-NOT: br
// SW-NOT: phi
// NATIVE: call i32 @llvm.x86.bmi.pdep.32(i32 %{{.*}}, i32 {{.*}}1431655765)
// CHECK: ret i32

uniform uint32 morton32(uniform uint32 x) { return pdep32(x, 0x55555555u); }
