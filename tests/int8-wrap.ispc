#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float bb) {
    #pragma ignore warning(perf)
    unsigned int8 a = aFOO[programIndex], b = bb;
    #pragma ignore warning(perf)
    RET[programIndex] = ((unsigned int8)100*a)+b; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (((100*(programIndex+1))&0xff)+5)&0xff;
}
