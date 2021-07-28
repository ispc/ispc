#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float aa = aFOO[programIndex]; 
    float<4> a = { -1, -2, -3, -4 };
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        a[3-programIndex] = aa;
    }
    RET[programIndex] = a[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = -3;
    RET[1] = 2;
}
