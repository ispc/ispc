//; RUN: %{ispc} --emit-asm --target=sse2-i32x4 --cpu=core2 %s -o - | FileCheck %s -check-prefix=CHECK_CPU_CORE2
//; RUN: %{ispc} --emit-asm --target=sse2-i32x4 --cpu=x86-64 %s -o - | FileCheck %s -check-prefix=CHECK_CPU_X86-64
//; RUN: %{ispc} --emit-asm --target=sse2-i32x8 --cpu=core2 %s -o - | FileCheck %s -check-prefix=CHECK_CPU_CORE2
//; RUN: %{ispc} --emit-asm --target=sse2-i32x8 --cpu=x86-64 %s -o - | FileCheck %s -check-prefix=CHECK_CPU_X86-64

//; REQUIRES: X86_ENABLED
int foo(uniform float b){
    int8 aa = b;
//; CHECK_CPU_CORE2: pshufb
//; CHECK_CPU_X86-64-NOT: pshufb
    return aa;
}
