#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programCount == 1)
        RET[0] = 3;
    else {
        int8 aa = aFOO[programIndex]; 
        int8 bb = aa + programCount;
        int8 shuf = shuffle(aa, bb, programIndex + 2);
        RET[programIndex] = shuf;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3 + programIndex;
}
