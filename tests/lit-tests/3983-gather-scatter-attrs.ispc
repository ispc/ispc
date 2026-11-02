// Regression test for https://github.com/ispc/ispc/issues/3983
// The base pointer of the x86 gathers and scatters used by the AVX2 and AVX-512
// builtins is marked as not captured, and the AVX-512 scatters, which have no
// memory attributes in LLVM, as only writing memory. They are not marked as
// only accessing memory through the base pointer: the offsets can hold the
// distance to another object, so the gathers in @relative_base stay apart.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX2
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX512
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=REL2
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=REL512
// REQUIRES: X86_ENABLED

// AVX2: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> {{[^,]*}}, ptr {{[^,]*}}{{nocapture|captures\(none\)}}

// AVX512-DAG: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(<16 x float> {{[^,]*}}, ptr {{[^,]*}}{{nocapture|captures\(none\)}}
// AVX512-DAG: call void @llvm.x86.avx512.mask.scatter.dps.512(ptr {{[^,]*}}{{nocapture|captures\(none\)}} {{[^)]*}}) #[[SC:[0-9]+]]
// AVX512: attributes #[[SC]] = { {{.*}}memory(write)

// REL2-LABEL: define void @relative_base(
// REL2-COUNT-4: @llvm.x86.avx2.gather
// REL2: {{^}}}

// REL512-LABEL: define void @relative_base(
// REL512-COUNT-4: @llvm.x86.avx512.mask.gather
// REL512: {{^}}}

noinline float gather(uniform float src[], int idx) { return src[idx]; }
noinline void scatter(uniform float dst[], int idx, float v) { dst[idx] = v; }

export void gather_scatter(uniform float src[], uniform float dst[], uniform int idx[]) {
    if (src[programIndex] > 0)
        scatter(dst, idx[programIndex], gather(src, idx[programIndex]));
}

uniform float a[16];
uniform float b[16];

// `relative` can hold the distance from a to b, so the second load must be done
// again after the store to b.
export void relative_base(uniform int64 relative[], uniform float out[]) {
    uniform float *varying p = (uniform float *varying)((uniform int64)&a[0] + relative[programIndex]);
    float x = *p;
    b[programIndex] = 5.0f;
    out[programIndex] = x + *p;
}
