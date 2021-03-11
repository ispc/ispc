//; RUN: %{ispc} %s --target=avx2-i32x4 --enable-intrinsic-call
//; RUN: not %{ispc} %s --target=sse2-i32x4 --enable-intrinsic-call --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_SSE2

//; REQUIRES: X86_ENABLED

//; CHECK_SSE2: Error: LLVM intrinsic "llvm.x86.avx.addsub.ps.256" not
float foo(float a, float b) {
    float c = @llvm.x86.avx.addsub.ps.256(a, b);
    return c;
}
