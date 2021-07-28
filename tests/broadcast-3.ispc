#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 a = aFOO[programIndex]; 
    int8 br = (programCount == 1) ? 4 : broadcast(a, (uniform int)b-2);
    RET[programIndex] = br;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
