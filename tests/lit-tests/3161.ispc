// Check that the 8-wide AVX-512 targets keep 8-wide vectors of 64-bit values in zmm
// registers by default, while 32-bit values stay in ymm registers, and that
// --opt=disable-zmm restores ymm-only code generation.

// RUN: %{ispc} %s --target=avx512skx-x8 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_512
// RUN: %{ispc} %s --target=avx512icl-x8 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_512
// RUN: %{ispc} %s --target=avx512skx-x8 --opt=disable-zmm --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s --target=avx512icl-x8 --opt=disable-zmm --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM
// RUN: %{ispc} %s -O2 --target=avx512icl-x8 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// RUN: %{ispc} %s -O2 --target=avx512icl-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"

// REQUIRES: X86_ENABLED

// CHECK_ATTR_512: attributes #{{[0-9]+}} = {{.*}}"min-legal-vector-width"="512"{{.*}}"prefer-vector-width"="512"
// CHECK_ATTR_512-NOT: "prefer-vector-width"="256"
// CHECK_ATTR_256: attributes #{{[0-9]+}} = {{.*}}"min-legal-vector-width"="256"{{.*}}"prefer-vector-width"="256"
// CHECK_ATTR_256-NOT: "prefer-vector-width"="512"

// CHECK_ZMM-LABEL: {{^}}scale_double:
// CHECK_ZMM: vmulpd {{.*}}%zmm
// CHECK_ZMM-LABEL: {{^}}scale_float:
// CHECK_ZMM-NOT: zmm
// CHECK_ZMM: vmulps {{.*}}%ymm
// CHECK_ZMM-NOT: zmm
// CHECK_ZMM: .Lfunc_end

export void scale_double(uniform double a[], uniform double s, uniform int n) {
    foreach (i = 0 ... n) {
        a[i] = a[i] * s;
    }
}

export void scale_float(uniform float a[], uniform float s, uniform int n) {
    foreach (i = 0 ... n) {
        a[i] = a[i] * s;
    }
}
