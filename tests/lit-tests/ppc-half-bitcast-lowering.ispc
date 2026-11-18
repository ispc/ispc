// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// REQUIRES: PPC64_ENABLED, LINUX_ENABLED
// RUN: %{ispc} %s --target=vsx-i32x4 --arch=ppc64le --target-os=linux --nostdlib --nowrap -O0 --emit-llvm-text -o - | FileCheck %s --implicit-check-not="call {{.*}}@llvm.ispc.bitcast" --implicit-check-not="@llvm.fabs." --implicit-check-not="@llvm.is.fpclass." --implicit-check-not="fcmp {{.*}}half"
// RUN: %{ispc} %s --target=vsx-i32x4 --arch=ppc64le --target-os=linux --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --implicit-check-not="call {{.*}}@llvm.ispc.bitcast" --implicit-check-not="@llvm.fabs." --implicit-check-not="@llvm.is.fpclass." --implicit-check-not="fcmp {{.*}}half"
// RUN: %{ispc} %s --target=vsx-i32x4 --arch=ppc64le --target-os=linux --nostdlib --nowrap -O2 --off-phase=11 --emit-llvm-text -o - | FileCheck %s --implicit-check-not="call {{.*}}@llvm.ispc.bitcast" --implicit-check-not="@llvm.fabs." --implicit-check-not="@llvm.is.fpclass." --implicit-check-not="fcmp {{.*}}half"
// RUN: %{ispc} %s --target=vsx-i32x4 --arch=ppc64le --target-os=linux --nostdlib --nowrap -O2 --off-phase=pre:last --emit-llvm-text -o - | FileCheck %s --implicit-check-not="call {{.*}}@llvm.ispc.bitcast" --implicit-check-not="@llvm.fabs." --implicit-check-not="@llvm.is.fpclass." --implicit-check-not="fcmp {{.*}}half"

// The backend cannot reliably lower half arithmetic on PowerPC. Keep these
// payload operations as integer operations, including when optimization is
// disabled, without leaving compiler-internal bitcast calls in the output.
// CHECK: and i16 {{.*}}, 32767
constexpr uniform float16 clear_sign(uniform float16 value) {
    uniform uint16 bits = @llvm.ispc.bitcast(value, (uniform uint16)0);
    return @llvm.ispc.bitcast(bits & (uniform uint16)0x7fff, (uniform float16)0);
}

uniform float16 runtime_clear_sign(uniform float16 value) { return clear_sign(value); }

// CHECK-LABEL: @finite_half
// CHECK: and <4 x i16>
varying bool finite_half(varying float16 value) {
    varying uint16 bits = @llvm.ispc.bitcast(value, (varying uint16)0);
    return (bits & (varying uint16)0x7fff) < (varying uint16)0x7c00;
}

// Frontend evaluation remains independent of the delayed runtime lowering.
static_assert(@llvm.ispc.bitcast(clear_sign(@llvm.ispc.bitcast((uniform uint16)0xfc35, (uniform float16)0)),
                               (uniform uint16)0) == 0x7c35);
