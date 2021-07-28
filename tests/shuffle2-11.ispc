#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int16 aa = aFOO[programIndex]; 
    int16 bb = aa + programCount;
    int16 shuf = shuffle(aa, bb, 2*programIndex);
    RET[programIndex] = shuf;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + 2*programIndex;
}
