#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    for (int i = 0; i < 16; ++i)
        sum += aFOO[i];
    RET[programIndex] = extract(sum, 0);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 136;
}
