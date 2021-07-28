#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programCount == 1)
        RET[0] = 3;
    else {
        int16 aa = aFOO[programIndex]; 
        int16 bb = aa + programCount;
        int16 shuf = shuffle(aa, bb, programCount + 1);
        RET[programIndex] = shuf;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2 + programCount;
}
