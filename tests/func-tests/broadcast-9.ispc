#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uint64 a = aFOO[programIndex]; 
    uint64 br = (programCount == 1) ? 4 : broadcast(a, (uniform int)b-2);
    RET[programIndex] = br;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
