#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uint8 aa = aFOO[programIndex]; 
    uint8 bb = aa + programCount;
    uint8 shuf = shuffle(aa, bb, 1);
    RET[programIndex] = shuf;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
