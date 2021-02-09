//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_WARNING

//; CHECK_WARNING: 21:19: Warning: Typecasting to type "/*unbound*/ float"
//; CHECK_WARNING: 25:19: Warning: Typecasting to type "/*unbound*/ float"
//; CHECK_WARNING: 29:26: Warning: Typecasting to type "/*unbound*/ float"
//; CHECK_WARNING: 33:23: Warning: Typecasting to type "/*unbound*/ float"

//; CHECK_WARNING-NOT: 38:
//; CHECK_WARNING-NOT: 39:
//; CHECK_WARNING-NOT: 40:

noinline float foo(float A, float B) {
    return A * B;
}

noinline float foo(float A, uniform float B) {
    return A * B;
}

noinline float bar(float A, uniform int B) {
    return foo(A, (float)B);
}

noinline float bar1(float A, uniform int B) {
    return foo(A, (float)B + (float)0);
}

noinline float bar2(float A, uniform int B, uniform bool cond) {
    return foo(A, cond ? (float)B : (float)0);
}

noinline float bar3(float A, uniform int B) {
    return foo(A, B + (float)0);
}

// Negative test
noinline float bar4(float A, uniform int B) {
    A = A + (float)B;
    B = B + (int)1;
    return foo(A, (varying float)B + (float)0);
}
