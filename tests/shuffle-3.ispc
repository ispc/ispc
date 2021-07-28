#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programCount == 1)
        RET[0] = 2;
    else {
        int8 a = aFOO[programIndex]; 
        int8 shuf = shuffle(a, 1);
        RET[programIndex] = shuf;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
