//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
//; CHECK: Warning: Converting from const pointer type "const uniform float * uniform" to pointer type "uniform float *

/* CASE 1 : const uniform float* uniform -> uniform float* uniform
   Print warning and do type conversion.*/
void foo_callee1(uniform float *uniform a) {
    return;
}

void fpp_caller1(const uniform float *uniform src) {
    foo_callee1(src);
    return;
}

/* CASE 2 : uniform float* const uniform -> uniform float* uniform
   Valid type conversion.*/
void foo_callee2(uniform float *uniform a) {
    return;
}
void fpp_caller2(uniform float *const uniform src) {
    foo_callee2(src);
    return;
}

/* CASE 3 : uniform float* uniform -> const uniform float* uniform
   Valid type conversion.*/
void foo_callee3(const uniform float *uniform a) {
    return;
}
void fpp_caller3(uniform float *uniform src) {
    foo_callee2(src);
    return;
}

/* CASE 4 : uniform float* uniform -> uniform float* const uniform
   Valid type conversion.*/
void foo_callee4(uniform float *const uniform a) {
    return;
}
void fpp_caller4(uniform float *uniform src) {
    foo_callee2(src);
    return;
}
