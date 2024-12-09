// The test checks correct features for ARM/AARCH64 for different cpus on Linux starting LLVM18. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a520 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A520

// arm supported on Linux only
// REQUIRES: ARM_ENABLED && LINUX_HOST && LLVM_18_0+

// CHECK_AARCH64_A520: Triple: aarch64-unknown-linux-gnu
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