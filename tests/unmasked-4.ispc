#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = aFOO[programIndex];
    if (a == 2) {
        unmasked {
            a = 0;
            if (b == 5)
                a = 1;
        }
    }
    RET[programIndex] = a;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[4] = 1;
}
