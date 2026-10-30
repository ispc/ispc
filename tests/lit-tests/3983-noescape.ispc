// Regression test for https://github.com/ispc/ispc/issues/3983
// The gather builtins don't capture their base pointer, so their declarations
// carry noescape and arrays that are only gathered from don't look captured.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --opt=disable-handle-pseudo-memory-ops --emit-llvm-text -o - | FileCheck %s --check-prefix=DECL
// RUN: %{ispc} %s --target=sse4-i32x4 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=SIG
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=SIG
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=SIG
// REQUIRES: X86_ENABLED

// DECL: declare <8 x float> @__pseudo_gather_{{(factored_)?}}base_offsets32_float(ptr {{[^,]*}}{{nocapture|captures\(none\)}}

// SIG: define void @gather_src(ptr noalias {{[^,]*}}{{nocapture|captures\(none\)}}{{[^,]*}} %src, ptr noalias{{[^,]*}} %idx, ptr noalias{{[^,]*}} %dst, i32 %n)
export void gather_src(uniform float src[], uniform int idx[], uniform float dst[], uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = src[idx[i]];
    }
}
