#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RNGState state;
    seed_rng(&state, programIndex);
    float sum = 0;
    uniform int iters = 40000;
    for (unsigned int i = 0; i < iters; ++i)
        sum += frandom(&state);
    float avg = sum / iters;
    RET[programIndex] = (avg > 0.495 && avg < 0.505) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
