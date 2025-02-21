// Check argument type deduction for template functions with short vector types

// RUN: %{ispc} -O2 --target=avx512icl-x4 --nowrap --nostdlib --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: Error: Unable to find any matching overload for call to function

uniform int foo(uniform int a, uniform int b) { return a + b; }
varying int foo(varying int a, varying int b) { return a + b; }

template <typename T, uint N> uniform T<N> foo(uniform T<N> a, uniform T<N> b) {
    uniform T<N> result;
    foreach (i = 0 ... N) {
        result[i] = foo(a[i], b[i]);
    }
    return result;
}

// CHECK-LABEL: @bar___{{.*}}(<4 x i32> %a, <4 x i32> %b, {{.*}}
// CHECK-DAG:   [[RET:%.*]] = add nsw <4 x i32> %b, %a
// CHECK-DAG:   ret <4 x i32> [[RET]]
uniform int<4> bar(uniform int<4> a, uniform int<4> b) {
    return foo(a, b);
}
