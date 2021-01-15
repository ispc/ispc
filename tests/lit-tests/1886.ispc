//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_WARNING

//; CHECK_WARNING: Warning: Typecasting to type "/*unbound*/ float" (variability not specified) from type

noinline float foo(float A, float B) {
    return A * B;
}

noinline float foo(float A, uniform float B) {
    return foo(A, (float)B);
}
