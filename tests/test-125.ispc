#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    if (a < 3) {
        if (all(a < 256)) {
            RET[programIndex] = 1;
        }
    }
    else RET[programIndex] = 0;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = RET[1] = 1;
}
