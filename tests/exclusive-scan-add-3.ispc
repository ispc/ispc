#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    float a = aFOO[programIndex];
    if (a <= 2)
        RET[programIndex] = exclusive_scan_add(a);
}

task void result(uniform float RET[]) {
    uniform int result[] = { 0, 1, 3 };
    RET[programIndex] = -1;
    if (programIndex <= 1) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = result[programIndex];
    }
}
