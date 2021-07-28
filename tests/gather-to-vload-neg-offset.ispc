#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    int index = programIndex + 5;
    #pragma ignore warning(perf)
    RET[index-b] = a; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
