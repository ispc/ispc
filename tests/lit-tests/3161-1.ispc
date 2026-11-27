// Check that --opt=disable-zmm also removes all zmm registers on the 8-wide targets that inherit
// their 64-bit builtins from avx512skx-x8 (avx512spr-x8, avx512gnr-x8, avx10.2dmr-x8; avx10.2nvl-x8
// is checked in 3161-2.ispc), and that these targets use zmm registers for 64-bit types by default.

// RUN: %{ispc} %S/3161.ispc -O2 --target=avx512spr-x8 --nowrap --emit-asm -o - | FileCheck %S/3161.ispc --check-prefix=CHECK_ZMM
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx512gnr-x8 --nowrap --emit-asm -o - | FileCheck %S/3161.ispc --check-prefix=CHECK_ZMM
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx10.2dmr-x8 --nowrap --emit-asm -o - | FileCheck %S/3161.ispc --check-prefix=CHECK_ZMM
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx512spr-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx512gnr-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// RUN: %{ispc} %S/3161.ispc -O2 --target=avx10.2dmr-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

// CHECK: {{^}}shuffle_double___{{.*}}:
