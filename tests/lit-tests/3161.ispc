// Check that the 8-wide AVX-512 targets keep 8-wide vectors of 64-bit values in zmm
// registers by default, while 32-bit values stay in ymm registers, and that
// --opt=disable-zmm restores ymm-only code generation.
// The 64-bit builtins of the zmm variant must use a single zmm instruction, not two ymm halves.

// RUN: %{ispc} %s --target=avx512skx-x8 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_512
// RUN: %{ispc} %s --target=avx512icl-x8 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_512
// RUN: %{ispc} %s --target=avx512skx-x8 --opt=disable-zmm --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s --target=avx512icl-x8 --opt=disable-zmm --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s --target=avx512skx-x8-nozmm --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_ATTR_256
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM
// RUN: %{ispc} %s -O2 --target=avx512icl-x8 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_YMM --implicit-check-not "zmm"
// RUN: %{ispc} %s -O2 --target=avx512icl-x8 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// RUN: %{ispc} %s -O2 --target=avx512skx-x8-nozmm --nowrap --emit-asm -o - | FileCheck %s --implicit-check-not "zmm"
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --addressing=64 --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ZMM
// RUN: %{ispc} %s -O2 --target=avx512skx-x8 --addressing=64 --opt=disable-zmm --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CHECK_YMM --implicit-check-not "zmm"

// REQUIRES: X86_ENABLED

// CHECK_ATTR_512: attributes #{{[0-9]+}} = {{.*}}"min-legal-vector-width"="512"{{.*}}"prefer-vector-width"="512"
// CHECK_ATTR_512-NOT: "prefer-vector-width"="256"
// CHECK_ATTR_256: attributes #{{[0-9]+}} = {{.*}}"min-legal-vector-width"="256"{{.*}}"prefer-vector-width"="256"
// CHECK_ATTR_256-NOT: "prefer-vector-width"="512"

// CHECK_ZMM-LABEL: {{^}}scale_double___{{.*}}:
// CHECK_ZMM: vmulpd {{.*}}%zmm
// CHECK_ZMM-LABEL: {{^}}scale_float___{{.*}}:
// CHECK_ZMM-NOT: zmm
// CHECK_ZMM: vmulps {{.*}}%ymm
// CHECK_ZMM-NOT: zmm
void scale_double(uniform double a[], uniform double s, uniform int n) {
    foreach (i = 0 ... n) {
        a[i] = a[i] * s;
    }
}

void scale_float(uniform float a[], uniform float s, uniform int n) {
    foreach (i = 0 ... n) {
        a[i] = a[i] * s;
    }
}

// CHECK_ZMM-LABEL: {{^}}shuffle_double___{{.*}}:
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4|vshuf[fi]64x2}}
// CHECK_ZMM: {{vpermq|vpermpd}} {{.*}}%zmm
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4|vshuf[fi]64x2}}
void shuffle_double(uniform double a[], uniform int perm[], uniform double r[]) {
    r[programIndex] = shuffle(a[programIndex], perm[programIndex]);
}

// CHECK_ZMM-LABEL: {{^}}shuffle2_int64___{{.*}}:
// CHECK_ZMM: {{vpermi2q|vpermt2q}} {{.*}}%zmm
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4}}
void shuffle2_int64(uniform int64 a[], uniform int64 b[], uniform int perm[], uniform int64 r[]) {
    r[programIndex] = shuffle(a[programIndex], b[programIndex], perm[programIndex]);
}

// CHECK_ZMM-LABEL: {{^}}masked_load_store_double___{{.*}}:
// CHECK_ZMM-NOT: kshiftrb
// CHECK_ZMM: vmovupd {{.*}}, %zmm{{[0-9]+}} {%k{{[0-9]}}} {z}
// CHECK_ZMM-NOT: kshiftrb
// CHECK_ZMM: vmovupd %zmm{{[0-9]+}}, {{.*}} {%k{{[0-9]}}}
// CHECK_ZMM-NOT: kshiftrb
void masked_load_store_double(uniform double a[], uniform double r[], uniform int n) {
    foreach (i = 0 ... n) {
        r[i] = a[i] + 1.0d;
    }
}

// CHECK_ZMM-LABEL: {{^}}gather_scatter_int64___{{.*}}:
// CHECK_ZMM-COUNT-1: vpgather{{[dq]}}q {{.*}}%zmm
// CHECK_ZMM-NOT: vpgather{{[dq]}}q
// CHECK_ZMM-COUNT-1: vpscatter{{[dq]}}q %zmm
// CHECK_ZMM-NOT: vpscatter{{[dq]}}q
void gather_scatter_int64(uniform int64 a[], uniform int idx[], uniform int64 r[]) {
    #pragma ignore warning(perf)
    r[idx[programIndex]] = a[idx[programIndex]];
}

// CHECK_ZMM-LABEL: {{^}}math_double___{{.*}}:
// CHECK_ZMM-DAG: vminpd {{.*}}%zmm
// CHECK_ZMM-DAG: vsqrtpd {{.*}}%zmm
// CHECK_ZMM-DAG: vrcp14pd {{.*}}%zmm
// CHECK_ZMM-DAG: vrsqrt14pd {{.*}}%zmm
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4}}
void math_double(uniform double a[], uniform double b[], uniform double r[]) {
    double x = a[programIndex], y = b[programIndex];
    r[programIndex] = min(x, y) + sqrt(x) + rcp_fast(y) + rsqrt_fast(x);
}

// CHECK_ZMM-LABEL: {{^}}minmax_int64___{{.*}}:
// CHECK_ZMM-DAG: vpminsq {{.*}}%zmm
// CHECK_ZMM-DAG: vpmaxuq {{.*}}%zmm
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4}}
void minmax_int64(uniform int64 a[], uniform int64 b[], uniform uint64 c[], uniform uint64 d[],
                         uniform int64 r[], uniform uint64 s[]) {
    r[programIndex] = min(a[programIndex], b[programIndex]);
    s[programIndex] = max(c[programIndex], d[programIndex]);
}

// CHECK_ZMM-LABEL: {{^}}math_double_full___{{.*}}:
// CHECK_ZMM-DAG: vrcp14pd {{.*}}%zmm
// CHECK_ZMM-DAG: vrsqrt14pd {{.*}}%zmm
// CHECK_ZMM-DAG: vfpclasspd {{.*}}%zmm
// CHECK_ZMM-NOT: {{vinsert[fi]64x4|vextract[fi]64x4}}
void math_double_full(uniform double a[], uniform double b[], uniform double r[]) {
    r[programIndex] = rcp(a[programIndex]) + rsqrt(b[programIndex]);
}

// The ABI of non-exported functions: a varying double is passed in one zmm register by default
// and in two ymm registers with --opt=disable-zmm.
// CHECK_ZMM-LABEL: {{^}}abi_add___{{.*}}:
// CHECK_ZMM-NEXT: # %bb.0:
// CHECK_ZMM-NEXT: vaddpd %zmm1, %zmm0, %zmm0
// CHECK_YMM-LABEL: {{^}}abi_add___{{.*}}:
// CHECK_YMM-NEXT: # %bb.0:
// CHECK_YMM-NEXT: vaddpd %ymm2, %ymm0, %ymm0
// CHECK_YMM-NEXT: vaddpd %ymm3, %ymm1, %ymm1
noinline double abi_add(double x, double y) { return x + y; }
