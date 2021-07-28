#include "../test_static.isph"
uniform int foo(int i);

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        val[i] = 0;

    foreach (i = 2 ... programCount)
        val[i] += i;

    uniform float sum = 0;
    for (uniform int i = 0; i < programCount; ++i) {
        sum += val[i];
    }

    RET[programIndex] = sum; 
}

task void result(uniform float RET[]) {
    int pi = (programIndex >= 2) ? programIndex : 0;
    RET[programIndex] = reduce_add(pi);
}
