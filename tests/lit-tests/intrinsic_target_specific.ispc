//; RUN: %{ispc} %s --target=avx2-i32x8 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --cpu=ps4 --target=avx1-i32x8 --enable-llvm-intrinsics
//; RUN: not %{ispc} %s --target=sse2-i32x8 --enable-llvm-intrinsics --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_SSE2

//; REQUIRES: X86_ENABLED

//; CHECK_SSE2: Error: Target specific LLVM intrinsic "avx.addsub.ps.256" not supported
float foo(float a, float b) {
    float c = @llvm.x86.avx.addsub.ps.256(a, b);
    return c;
}
