#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    float a = aFOO[programIndex];
    foreach_unique (ua in a)
        ++count;
    RET[programIndex] = count; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount;
}
