// Generic targets must not report SSE2 or the specialized RVV target.
// RUN: %{ispc} %s --arch=riscv64 --cpu=generic-rv64gcv --target-os=linux --target=generic-i32x8 --nostdlib --nowrap -E -dM -o - | FileCheck %s --check-prefix=GENERIC --implicit-check-not=ISPC_TARGET_SSE2 --implicit-check-not=ISPC_TARGET_RV64GCV
// RUN: %{ispc} %s --arch=riscv64 --cpu=generic-rv64gcv --target-os=linux --target=generic-i16x16 --nostdlib --nowrap -E -dM -o - | FileCheck %s --check-prefix=GENERIC --implicit-check-not=ISPC_TARGET_SSE2 --implicit-check-not=ISPC_TARGET_RV64GCV
// RUN: %{ispc} %s --arch=riscv64 --cpu=generic-rv64gcv --target-os=linux --target=rvv-x4 --nostdlib --nowrap -E -dM -o - | FileCheck %s --check-prefix=RVV --implicit-check-not=ISPC_TARGET_GENERIC --implicit-check-not=ISPC_TARGET_SSE2
// REQUIRES: RISCV_ENABLED && LINUX_ENABLED

// GENERIC: #define ISPC_TARGET_GENERIC 1
// RVV: #define ISPC_TARGET_RV64GCV 1
