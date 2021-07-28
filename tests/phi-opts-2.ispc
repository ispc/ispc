#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    for (int i = programIndex; i < 16; i += programCount)
        sum += aFOO[i];
    RET[programIndex] = reduce_add(sum); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 136;
}
