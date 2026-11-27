// Check that with --math-lib=svml and --opt=disable-zmm, the 16-wide AVX-512 targets call the ymm
// SVML entry points (8 x float, 4 x double) and don't pass ymm register pairs to the zmm entry points.

// RUN: %{ispc} %s -O2 --target=avx512skx-x16 --opt=disable-zmm --math-lib=svml --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_YMM --implicit-check-not "%zmm"
// RUN: %{ispc} %s -O2 --target=avx512icl-x16 --opt=disable-zmm --math-lib=svml --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_YMM --implicit-check-not "%zmm"
// RUN: %{ispc} %s -O2 --target=avx512skx-x16-nozmm --math-lib=svml --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_YMM --implicit-check-not "%zmm"
// RUN: %{ispc} %s -O2 --target=avx512skx-x16 --math-lib=svml --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM

// REQUIRES: X86_ENABLED && LINUX_ENABLED

// CHECK_YMM-LABEL: {{^}}exp_double___{{.*}}:
// CHECK_YMM-COUNT-4: {{callq?|jmp}} {{.*}}__svml_exp4_z0
// CHECK_YMM-NOT: __svml_exp
// CHECK_ZMM-LABEL: {{^}}exp_double___{{.*}}:
// CHECK_ZMM-COUNT-2: {{callq?|jmp}} {{.*}}__svml_exp8_z0
// CHECK_ZMM-NOT: __svml_exp
double exp_double(double x) { return exp(x); }

// CHECK_YMM-LABEL: {{^}}exp_float___{{.*}}:
// CHECK_YMM-COUNT-2: {{callq?|jmp}} {{.*}}__svml_expf8_z0
// CHECK_YMM-NOT: __svml_exp
// CHECK_ZMM-LABEL: {{^}}exp_float___{{.*}}:
// CHECK_ZMM-COUNT-1: {{callq?|jmp}} {{.*}}__svml_expf16_z0
// CHECK_ZMM-NOT: __svml_exp
float exp_float(float x) { return exp(x); }
