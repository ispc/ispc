// Same as 3161-1.ispc for avx10.2nvl-x8, which requires LLVM 22 or later.

// RUN: %{ispc} %S/3161.ispc -O2 --target=avx10.2nvl-x8 --nowrap --emit-asm -o - | FileCheck %S/3161.ispc --check-prefix=CHECK_ZMM
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx10.2nvl-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+

// CHECK: {{^}}shuffle_double___{{.*}}:
