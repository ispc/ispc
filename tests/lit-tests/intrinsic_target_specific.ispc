//; RUN: %{ispc} %s --target=avx2-i32x8 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --cpu=ps4 --target=avx1-i32x8 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --nowrap --cpu=spr --target=avx512spr-x8 --enable-llvm-intrinsics
//; RUN: %{ispc} %s --nowrap --target=avx512gnr-x8 --enable-llvm-intrinsics
//; RUN: not %{ispc} %s --target=sse2-i32x8 --enable-llvm-intrinsics --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_SSE2
//; RUN: %{ispc} %s -DFP16 --nowrap --cpu=spr --target=avx512spr-x8 --enable-llvm-intrinsics --emit-asm -o - | FileCheck %s -check-prefix=CHECK_FP16
//; RUN: %{ispc} %s -DFP16 --nowrap --cpu=gnr --target=avx512gnr-x8 --enable-llvm-intrinsics --emit-asm -o - | FileCheck %s -check-prefix=CHECK_FP16
//; RUN: not %{ispc} %s -DFP16 --nowrap --cpu=icl --target=avx512icl-x8 --enable-llvm-intrinsics -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_FP16_ICL

//; REQUIRES: X86_ENABLED && !MACOS_HOST

//; CHECK_SSE2: Error: Target specific LLVM intrinsic "avx.addsub.ps.256" not supported
float foo(float a, float b) {
    float c = @llvm.x86.avx.addsub.ps.256(a, b);
    return c;
}

#ifdef FP16
//; CHECK_FP16: vmaxph
//; CHECK_FP16_ICL: Error: Target specific LLVM intrinsic "avx512fp16.max.ph.128" not supported on "icelake-client" CPU.
float16 bar(float16 a, float16 b) { return @llvm.x86.avx512fp16.max.ph.128(a, b); }
#endif
