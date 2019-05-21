//; RUN: not %{ispc} %s --target=sse2-i32x4 2>&1 | FileCheck %s
//; CHECK: Error:
noinline inline noinline int bar(uniform int a[]) {
    return a[programIndex];
}

export  void foo(uniform int a[]) {
    a[programIndex] = bar(a);
}
