#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 a = aFOO[programIndex]; 
    int reverse = programCount - 1 - programIndex + (int)b - 5;
    int8 shuf = shuffle(a, reverse);
    RET[programIndex] = shuf;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
