#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 a = aFOO[programIndex]; 
    int8 rot = rotate(a, 2);
    RET[programIndex] = rot;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + (programIndex + 2) % programCount;
}
