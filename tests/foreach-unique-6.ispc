#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    float * ptr = &aFOO[programIndex];

    foreach_unique (p in ptr)
        ++count;

    RET[programIndex] = count; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount;
}
