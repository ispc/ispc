#include "../test_static.isph"
void inc(uniform float<4> * uniform v) { ++(*v); }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float<4> x[2] = { {b,b+1,b+2,b+3}, {1,2,3,4} };

    inc(&x[1]);
    RET[programIndex] = 0;
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = x[1][programIndex];
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 2;
    RET[1] = 3;
    RET[2] = 4;
    RET[3] = 5;
}
