// Check that gather/scatter instructions are generated properly for AVX2 and AVX512 targets.
// Not testing x86 vs x86-64, just choosing platform's defaultarch , because that would break platforms without x86.

// RUN: %{ispc} %s -O2 --woff --target=avx2-i64x4 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X4_32
// RUN: %{ispc} %s -O2 --woff --target=avx2-i64x4 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X4_64
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x4 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X4_32
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x4 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X4_64
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x8 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X8_32
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x8 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X8_64
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x16 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X16_32
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x16 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX2_X16_64
// avx2-i16x16 is missing gathers.
// avx2-i8x32 is missing gathers.
// RUN: %{ispc} %s -O2 --woff --target=avx512knl-i32x16 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X16_32
// RUN: %{ispc} %s -O2 --woff --target=avx512knl-i32x16 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X16_64
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x4 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X4_32
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x4 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X4_64
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x8 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X8_32
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x8 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X8_64
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x16 --addressing=32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X16_32
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-i32x16 --addressing=64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_AVX512_X16_64
// avx512skx-i8x64 is missing gathers and scatters.
// avx512skx-i32x32 is missing gathers and scatters.

// REQUIRES: X86_ENABLED

// CHECK_AVX2_X4_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X4_32-COUNT-1: vpgatherdd
// CHECK_AVX2_X4_32-NOT: vpgatherdd

// CHECK_AVX2_X4_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X4_64-COUNT-1: vpgatherqd
// CHECK_AVX2_X4_64-NOT: vpgatherqd

// CHECK_AVX2_X8_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X8_32-COUNT-1: vpgatherdd
// CHECK_AVX2_X8_32-NOT: vpgatherdd

// CHECK_AVX2_X8_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X8_64-COUNT-2: vpgatherqd
// CHECK_AVX2_X8_64-NOT: vpgatherqd

// CHECK_AVX2_X16_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X16_32-COUNT-2: vpgatherdd
// CHECK_AVX2_X16_32-NOT: vpgatherdd

// CHECK_AVX2_X16_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX2_X16_64-COUNT-4: vpgatherqd
// CHECK_AVX2_X16_64-NOT: vpgatherqd

// CHECK_AVX512_X4_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X4_32-COUNT-1: vpgatherdd
// CHECK_AVX512_X4_32-NOT: vpgatherdd

// CHECK_AVX512_X4_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X4_64-COUNT-1: vpgatherqd
// CHECK_AVX512_X4_64-NOT: vpgatherqd

// CHECK_AVX512_X8_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X8_32-COUNT-1: vpgatherdd
// CHECK_AVX512_X8_32-NOT: vpgatherdd

// CHECK_AVX512_X8_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X8_64-COUNT-2: vpgatherqd
// CHECK_AVX512_X8_64-NOT: vpgatherqd

// CHECK_AVX512_X16_32-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X16_32-COUNT-1: vpgatherdd
// CHECK_AVX512_X16_32-NOT: vpgatherdd

// CHECK_AVX512_X16_64-LABEL: gather_int32___un_3C_uni_3E_vyi:
// CHECK_AVX512_X16_64-COUNT-2: vpgatherqd
// CHECK_AVX512_X16_64-NOT: vpgatherqd
int32 gather_int32(uniform int32 p[], int i) {
  return p[i];
}

// CHECK_AVX2_X4_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X4_32-COUNT-1: vgatherdps
// CHECK_AVX2_X4_32-NOT: vgatherdps

// CHECK_AVX2_X4_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X4_64-COUNT-1: vgatherqps
// CHECK_AVX2_X4_64-NOT: vgatherqps

// CHECK_AVX2_X8_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X8_32-COUNT-1: vgatherdps
// CHECK_AVX2_X8_32-NOT: vgatherdps

// CHECK_AVX2_X8_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X8_64-COUNT-2: vgatherqps
// CHECK_AVX2_X8_64-NOT: vgatherqps

// CHECK_AVX2_X16_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X16_32-COUNT-2: vgatherdps
// CHECK_AVX2_X16_32-NOT: vgatherdps

// CHECK_AVX2_X16_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX2_X16_64-COUNT-4: vgatherqps
// CHECK_AVX2_X16_64-NOT: vgatherqps

// CHECK_AVX512_X4_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X4_32-COUNT-1: vgatherdps
// CHECK_AVX512_X4_32-NOT: vgatherdps

// CHECK_AVX512_X4_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X4_64-COUNT-1: vgatherqps
// CHECK_AVX512_X4_64-NOT: vgatherqps

// CHECK_AVX512_X8_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X8_32-COUNT-1: vgatherdps
// CHECK_AVX512_X8_32-NOT: vgatherdps

// CHECK_AVX512_X8_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X8_64-COUNT-2: vgatherqps
// CHECK_AVX512_X8_64-NOT: vgatherqps

// CHECK_AVX512_X16_32-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X16_32-COUNT-1: vgatherdps
// CHECK_AVX512_X16_32-NOT: vgatherdps

// CHECK_AVX512_X16_64-LABEL: gather_float___un_3C_unf_3E_vyi:
// CHECK_AVX512_X16_64-COUNT-2: vgatherqps
// CHECK_AVX512_X16_64-NOT: vgatherqps
float gather_float(uniform float p[], int i) {
  return p[i];
}

// avx2 doesn't have scatter instructions.
// CHECK_AVX2_X4_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX2_X4_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX2_X8_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX2_X8_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX2_X16_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX2_X16_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:

// CHECK_AVX512_X4_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X4_32-COUNT-1: vpscatterdd
// CHECK_AVX512_X4_32-NOT: vpscatterdd

// CHECK_AVX512_X4_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X4_64-COUNT-1: vpscatterqd
// CHECK_AVX512_X4_64-NOT: vpscatterqd

// CHECK_AVX512_X8_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X8_32-COUNT-1: vpscatterdd
// CHECK_AVX512_X8_32-NOT: vpscatterdd

// CHECK_AVX512_X8_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X8_64-COUNT-2: vpscatterqd
// CHECK_AVX512_X8_64-NOT: vpscatterqd

// CHECK_AVX512_X16_32-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X16_32-COUNT-1: vpscatterdd
// CHECK_AVX512_X16_32-NOT: vpscatterdd

// CHECK_AVX512_X16_64-LABEL: scatter_i32___un_3C_uni_3E_vyivyi:
// CHECK_AVX512_X16_64-COUNT-2: vpscatterqd
// CHECK_AVX512_X16_64-NOT: vpscatterqd
void scatter_i32(uniform int32 p[], int i, int32 v) {
  p[i] = v;
}

// avx2 doesn't have scatter instructions.
// CHECK_AVX2_X4_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX2_X4_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX2_X8_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX2_X8_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX2_X16_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX2_X16_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:

// CHECK_AVX512_X4_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X4_32-COUNT-1: vscatterdps
// CHECK_AVX512_X4_32-NOT: vscatterdps

// CHECK_AVX512_X4_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X4_64-COUNT-1: vscatterqps
// CHECK_AVX512_X4_64-NOT: vscatterqps

// CHECK_AVX512_X8_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X8_32-COUNT-1: vscatterdps
// CHECK_AVX512_X8_32-NOT: vscatterdps

// CHECK_AVX512_X8_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X8_64-COUNT-2: vscatterqps
// CHECK_AVX512_X8_64-NOT: vscatterqps

// CHECK_AVX512_X16_32-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X16_32-COUNT-1: vscatterdps
// CHECK_AVX512_X16_32-NOT: vscatterdps

// CHECK_AVX512_X16_64-LABEL: scatter_float___un_3C_unf_3E_vyivyf:
// CHECK_AVX512_X16_64-COUNT-2: vscatterqps
// CHECK_AVX512_X16_64-NOT: vscatterqps
void scatter_float(uniform float p[], int i, float v) {
  p[i] = v;
}

// CHECK_AVX2_X4_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X4_32-COUNT-1: vpgatherdq
// CHECK_AVX2_X4_32-NOT: vpgatherdq

// CHECK_AVX2_X4_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X4_64-COUNT-1: vpgatherqq
// CHECK_AVX2_X4_64-NOT: vpgatherqq

// CHECK_AVX2_X8_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X8_32-COUNT-2: vpgatherdq
// CHECK_AVX2_X8_32-NOT: vpgatherdq

// CHECK_AVX2_X8_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X8_64-COUNT-2: vpgatherqq
// CHECK_AVX2_X8_64-NOT: vpgatherqq

// CHECK_AVX2_X16_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X16_32-COUNT-4: vpgatherdq
// CHECK_AVX2_X16_32-NOT: vpgatherdq

// CHECK_AVX2_X16_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX2_X16_64-COUNT-4: vpgatherqq
// CHECK_AVX2_X16_64-NOT: vpgatherqq

// CHECK_AVX512_X4_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X4_32-COUNT-1: vpgatherdq
// CHECK_AVX512_X4_32-NOT: vpgatherdq

// CHECK_AVX512_X4_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X4_64-COUNT-1: vpgatherqq
// CHECK_AVX512_X4_64-NOT: vpgatherqq

// CHECK_AVX512_X8_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X8_32-COUNT-2: vpgatherdq
// CHECK_AVX512_X8_32-NOT: vpgatherdq

// CHECK_AVX512_X8_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X8_64-COUNT-2: vpgatherqq
// CHECK_AVX512_X8_64-NOT: vpgatherqq

// CHECK_AVX512_X16_32-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X16_32-COUNT-2: vpgatherdq
// CHECK_AVX512_X16_32-NOT: vpgatherdq

// CHECK_AVX512_X16_64-LABEL: gather_int64___un_3C_unI_3E_vyi:
// CHECK_AVX512_X16_64-COUNT-2: vpgatherqq
// CHECK_AVX512_X16_64-NOT: vpgatherqq
int64 gather_int64(uniform int64 p[], int i) {
  return p[i];
}

// CHECK_AVX2_X4_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X4_32-COUNT-1: vgatherdpd
// CHECK_AVX2_X4_32-NOT: vgatherdpd

// CHECK_AVX2_X4_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X4_64-COUNT-1: vgatherqpd
// CHECK_AVX2_X4_64-NOT: vgatherqpd

// CHECK_AVX2_X8_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X8_32-COUNT-2: vgatherdpd
// CHECK_AVX2_X8_32-NOT: vgatherdpd

// CHECK_AVX2_X8_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X8_64-COUNT-2: vgatherqpd
// CHECK_AVX2_X8_64-NOT: vgatherqpd

// CHECK_AVX2_X16_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X16_32-COUNT-4: vgatherdpd
// CHECK_AVX2_X16_32-NOT: vgatherdpd

// CHECK_AVX2_X16_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX2_X16_64-COUNT-4: vgatherqpd
// CHECK_AVX2_X16_64-NOT: vgatherqpd

// CHECK_AVX512_X4_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X4_32-COUNT-1: vgatherdpd
// CHECK_AVX512_X4_32-NOT: vgatherdpd

// CHECK_AVX512_X4_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X4_64-COUNT-1: vgatherqpd
// CHECK_AVX512_X4_64-NOT: vgatherqpd

// CHECK_AVX512_X8_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X8_32-COUNT-2: vgatherdpd
// CHECK_AVX512_X8_32-NOT: vgatherdpd

// CHECK_AVX512_X8_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X8_64-COUNT-2: vgatherqpd
// CHECK_AVX512_X8_64-NOT: vgatherqpd

// CHECK_AVX512_X16_32-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X16_32-COUNT-2: vgatherdpd
// CHECK_AVX512_X16_32-NOT: vgatherdpd

// CHECK_AVX512_X16_64-LABEL: gather_double___un_3C_und_3E_vyi:
// CHECK_AVX512_X16_64-COUNT-2: vgatherqpd
// CHECK_AVX512_X16_64-NOT: vgatherqpd
double gather_double(uniform double p[], int i) {
  return p[i];
}

// avx2 doesn't have scatter instructions.
// CHECK_AVX2_X4_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX2_X4_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX2_X8_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX2_X8_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX2_X16_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX2_X16_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:

// CHECK_AVX512_X4_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X4_32-COUNT-1: vpscatterdq
// CHECK_AVX512_X4_32-NOT: vpscatterdq

// CHECK_AVX512_X4_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X4_64-COUNT-1: vpscatterqq
// CHECK_AVX512_X4_64-NOT: vpscatterqq

// CHECK_AVX512_X8_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X8_32-COUNT-2: vpscatterdq
// CHECK_AVX512_X8_32-NOT: vpscatterdq

// CHECK_AVX512_X8_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X8_64-COUNT-2: vpscatterqq
// CHECK_AVX512_X8_64-NOT: vpscatterqq

// CHECK_AVX512_X16_32-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X16_32-COUNT-2: vpscatterdq
// CHECK_AVX512_X16_32-NOT: vpscatterdq

// CHECK_AVX512_X16_64-LABEL: scatter_i64___un_3C_unI_3E_vyivyI:
// CHECK_AVX512_X16_64-COUNT-2: vpscatterqq
// CHECK_AVX512_X16_64-NOT: vpscatterqq
void scatter_i64(uniform int64 p[], int i, int64 v) {
  p[i] = v;
}

// avx2 doesn't have scatter instructions.
// CHECK_AVX2_X4_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX2_X4_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX2_X8_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX2_X8_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX2_X16_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX2_X16_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:

// CHECK_AVX512_X4_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X4_32-COUNT-1: vscatterdpd
// CHECK_AVX512_X4_32-NOT: vscatterdpd

// CHECK_AVX512_X4_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X4_64-COUNT-1: vscatterqpd
// CHECK_AVX512_X4_64-NOT: vscatterqpd

// CHECK_AVX512_X8_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X8_32-COUNT-2: vscatterdpd
// CHECK_AVX512_X8_32-NOT: vscatterdpd

// CHECK_AVX512_X8_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X8_64-COUNT-2: vscatterqpd
// CHECK_AVX512_X8_64-NOT: vscatterqpd

// CHECK_AVX512_X16_32-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X16_32-COUNT-2: vscatterdpd
// CHECK_AVX512_X16_32-NOT: vscatterdpd

// CHECK_AVX512_X16_64-LABEL: scatter_double___un_3C_und_3E_vyivyd:
// CHECK_AVX512_X16_64-COUNT-2: vscatterqpd
// CHECK_AVX512_X16_64-NOT: vscatterqpd
void scatter_double(uniform double p[], int i, double v) {
  p[i] = v;
}
