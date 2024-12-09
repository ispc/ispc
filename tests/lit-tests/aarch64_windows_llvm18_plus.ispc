// The test checks correct features for AARCH64 for different cpus on Windows stating LLVM 18. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a520 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A520

// REQUIRES: ARM_ENABLED && WINDOWS_HOST && LLVM_18_0+

// CHECK_AARCH64_A520: Triple: aarch64-pc-windows-msvc
// CHECK_AARCH64_A520-NEXT: CPU: cortex-a520
// CHECK_AARCH64_A520-NEXT: Feature String:
// CHECK_AARCH64_A520-SAME: +crc
// CHECK_AARCH64_A520-SAME: +dotprod
// CHECK_AARCH64_A520-SAME: +fp-armv8
// CHECK_AARCH64_A520-SAME: +fp16fml
// CHECK_AARCH64_A520-SAME: +fullfp16
// CHECK_AARCH64_A520-SAME: +i8mm
// CHECK_AARCH64_A520-SAME: +lse
// CHECK_AARCH64_A520-SAME: +neon
// CHECK_AARCH64_A520-SAME: +rcpc
// CHECK_AARCH64_A520-SAME: +sve
// CHECK_AARCH64_A520-SAME: +sve2

uniform int i;

void foo() {}