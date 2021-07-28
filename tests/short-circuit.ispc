#include "../test_static.isph"


task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform float array[4] = {1,1,1,1};
    int index = 1000 * (a-1);
    #pragma ignore warning(perf)
    RET[programIndex] = (programIndex == 0) ? array[index] : 2.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
    RET[0] = 1;
}
