//; RUN: %{ispc} %s --target=sse2-i32x4 -h %t.h --emit-asm -o - | FileCheck %s
//; REQUIRES: X86_ENABLED
noinline int bar(uniform int a[]) {
    return a[programIndex];
}

export  void foo(uniform int a[]) {
// CHECK: callq	{{.?}}bar___un_3C_uni_3E
    a[programIndex] = bar(a);
}
