#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 10., t = 2., c;
    if (a <= t) {
        if (a == 1.) { RET[programIndex] = 7.; return; }
        else { RET[programIndex] = a; return; }
    }
    else {
        c = b;
        if (a == 3.) { RET[programIndex] = 12; return; }
    }
    RET[programIndex] = c;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 10;
    RET[0] = 7;
    RET[1] = 2;
    RET[2] = 12;
}
