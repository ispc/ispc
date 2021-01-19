//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_WARNING

//; CHECK_WARNING: 24:19: Warning: Typecasting to type "/*unbound*/ float"

//; CHECK_WARNING: 28:19: Warning: Typecasting to type "/*unbound*/ float"

//; CHECK_WARNING: 32:26: Warning: Typecasting to type "/*unbound*/ float"

//; CHECK_WARNING: 36:23: Warning: Typecasting to type "/*unbound*/ float"

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

noinline float bar4(float A, uniform int B) {
    return foo(A, (varying float)B + (float)0);
}
