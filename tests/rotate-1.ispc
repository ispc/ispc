#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int rot = rotate(a, -1);
    RET[programIndex] = rot;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + (programIndex + programCount - 1) % programCount;
}
