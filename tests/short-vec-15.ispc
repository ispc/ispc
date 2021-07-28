#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float<4> x = {a,a+1,a+2,a+3}, y = {a,a-1,a+2,0};

    RET[programIndex] = 0;

    x = (x < y) ? x : y;
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = x[programIndex];
    }

}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 1;
    RET[2] = 5;
    RET[3] = 0;
}
