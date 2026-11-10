// Check that LLVM intrinsics with scalable vector types (SVE) are reported as
// not representable instead of being mapped to uniform short vectors
// (https://github.com/ispc/ispc/issues/4004).

// RUN: not %{ispc} --target=neon-i32x4 --arch=aarch64 --nowrap --nostdlib -o %t.o %s 2>&1 | FileCheck %s --implicit-check-not=Error --implicit-check-not=FATAL

// REQUIRES: ARM_ENABLED

// CHECK: llvm-intrinsics-scalable.ispc:[[@LINE+1]]:36: Error: Return type "<vscale x 8 x half>" of LLVM intrinsic "llvm.aarch64.sve.fcvt.f16f32" is not representable as an ISPC type.
uniform float f(uniform float a) { @llvm.aarch64.sve.fcvt.f16f32(a, a, a); return a; }
