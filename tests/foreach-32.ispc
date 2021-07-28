#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
#define A_BEGIN 11
#define A_END   14
#define B_BEGIN 28
#define B_END   31
#define C_BEGIN 0
#define C_END   8
    uniform int t = 0;

    foreach_tiled (i = A_BEGIN ... A_END, j = B_BEGIN ... B_END, k = C_BEGIN ... C_END) {
        t++;
    }

    //the comparison with the expected number of iterations
    if (programCount == 4)
        RET[programIndex] = t - 24;
    else if (programCount == 8)
        RET[programIndex] = t - 16;
    else if (programCount == 16)
        RET[programIndex] = t - 8;
    else if (programCount == 32)
        RET[programIndex] = t - 4;
    else if (programCount == 64)
        RET[programIndex] = t - 2;
    else
        RET[programIndex] = t; //this case is still unknown, error in general
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
