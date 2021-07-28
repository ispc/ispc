#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float bb) {
    #pragma ignore warning(perf)
    unsigned int16 a = aFOO[programIndex], b = bb;
    #pragma ignore warning(perf)
    RET[programIndex] = ((unsigned int16)4000*a)+b; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (((4000*(programIndex+1))&0xffff)+5)&0xffff;
}
