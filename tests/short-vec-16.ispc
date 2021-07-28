#include "../test_static.isph"
void inc(uniform float<4> * uniform v) { ++(*v); }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float<4> x = {b,b+1,b+2,b+3}, y = {b,b-1,b+2,0};

    inc(&x);
    RET[programIndex] = 0;
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = x[programIndex];
    }

}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 6;
    RET[1] = 7;
    RET[2] = 8;
    RET[3] = 9;
}
