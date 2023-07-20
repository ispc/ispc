#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[programIndex + programCount] = 0;

    float f1 = 65535;
    unsigned int16 h1 = float_to_half_fast(f1);

    float f2 = 32768;
    unsigned int16 h2 = float_to_half_fast(f2);

    RET[programIndex] = half_to_float_fast(h1);
    RET[programIndex + programCount] = half_to_float_fast(h2);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 65535;
    RET[programIndex + programCount] = 32768;
}

