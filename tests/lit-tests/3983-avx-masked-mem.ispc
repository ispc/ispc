// Regression test for https://github.com/ispc/ispc/issues/3983
// The x86 masked load and store intrinsics used by the AVX1 and AVX2 builtins
// are marked as not capturing their pointer, so the arrays they access don't
// look captured, and they still select vmaskmov.

// RUN: %{ispc} %s --target=avx1-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx1-i32x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx1-i64x4 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx2-i32x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx2-i64x4 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} %s --target=avx1-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=CALL
// RUN: %{ispc} %s --target=avx1-i64x4 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=CALL
// RUN: %{ispc} %s --target=avx2-i32x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=CALL
// RUN: %{ispc} %s --target=avx1-i32x8 --nowrap --woff -O2 --emit-asm -o - | FileCheck %s --check-prefix=ASM
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --emit-asm -o - | FileCheck %s --check-prefixes=ASM,ASM2
// REQUIRES: X86_ENABLED

// IR: define void @copy_masked(ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %src, ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %dst, ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %dst64, i32 %n)
// IR: define void @copy_masked64(ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %src, ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %dst, i32 %n)
// IR: define {{(noundef )?}}i32 @pack(ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %dst, ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %dst64,

// With a mask that isn't known, the x86 intrinsics stay and carry the attributes.
// CALL-DAG: call {{.*}}@llvm.x86.avx.maskload.ps{{(.256)?}}(ptr {{[^,]*}}{{nocapture|captures\(none\)}}
// CALL-DAG: call {{.*}}@llvm.x86.avx.maskload.pd.256(ptr {{[^,]*}}{{nocapture|captures\(none\)}}
// CALL-DAG: call void @llvm.x86.avx.maskstore.ps{{(.256)?}}(ptr {{[^,]*}}{{nocapture|captures\(none\)}} {{[^)]*}}) #[[ST:[0-9]+]]
// CALL-DAG: call void @llvm.x86.avx.maskstore.pd.256(ptr {{[^,]*}}{{nocapture|captures\(none\)}} {{[^)]*}}) #[[ST]]
// CALL: attributes #[[ST]] = { {{.*}}memory(argmem: write)

// ASM-LABEL: copy_masked:
// ASM-DAG: vmaskmovps
// ASM-DAG: vmaskmovpd
// ASM-LABEL: copy_masked64:
// ASM: vmaskmovpd
// ASM-LABEL: pack:
// ASM: vmaskmov
// ASM2: vpmaskmovd %ymm{{[0-9]+}}, %ymm{{[0-9]+}}, (
export void copy_masked(uniform float src[], uniform float dst[], uniform double dst64[], uniform int n) {
    foreach (i = 0 ... n) {
        float v = src[i];
        if (v > 0) {
            dst[i] = v;
            dst64[i] = v;
        }
    }
}

export void copy_masked64(uniform double src[], uniform int64 dst[], uniform int n) {
    foreach (i = 0 ... n) {
        double v = src[i];
        if (v > 0)
            dst[i] = (int64)v;
    }
}

export uniform int pack(uniform int dst[], uniform int64 dst64[], uniform int src[], uniform int n) {
    uniform int k = 0;
    foreach (i = 0 ... n) {
        int v = src[i];
        if (v > 0) {
            packed_store_active2(&dst64[k], (int64)v);
            k += packed_store_active2(&dst[k], v);
        }
    }
    return k;
}

noinline void masked_st(uniform float dst[], float v) { dst[programIndex] = v; }
noinline void masked_st64(uniform double dst[], double v) { dst[programIndex] = v; }
noinline float masked_ld(uniform float src[]) { return src[programIndex]; }
noinline double masked_ld64(uniform double src[]) { return src[programIndex]; }

export void unknown_mask(uniform float a[], uniform double b[]) {
    if (a[programIndex] > 0) {
        masked_st(a, masked_ld(a) + 1);
        masked_st64(b, masked_ld64(b) + 1);
    }
}
