#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int<4> x = {-1,-2,3,4}, y = {b-5,b-5,b-5,b-5};

    RET[programIndex] = 0;

    int index = aFOO[programIndex];
    index = min(index, 3);

    x = (x < y) ? x : y;
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = x[programIndex];
    }

/*CO    return c[0] ? 1 : 0;*/
/*CO    x = b;*/
/*CO    y = b;*/
/*CO    return x+y;*/
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = -1;
    RET[1] = -2;
    RET[2] = 0;
    RET[3] = 0;
}
