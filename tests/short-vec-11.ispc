#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float<3> x = {1,2,3}, y = {b-5,b-5,b-5};

    RET[programIndex] = 0;

    uniform float<3> z = x[b-5];

    if (programIndex < 3) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = z[programIndex];
    }

/*CO    int index = aFOO[programIndex];*/
/*CO    index = min(index, 3);*/
/*CO    return x[index];*/

/*CO    return x << 1;*/
/*CO    return c[0] ? 1 : 0;*/
/*CO    x = b;*/
/*CO    y = b;*/
/*CO    return x+y;*/
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 1;
    RET[2] = 1;
}
