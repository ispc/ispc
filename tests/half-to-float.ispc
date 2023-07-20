#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[programIndex + programCount] = 0;

    unsigned int16 h1 = 65535;
    float f1 = half_to_float(h1);

    unsigned int16 h2 = 32768;
    float f2 = half_to_float(h2);

    RET[programIndex] = float_to_half(f1);
    RET[programIndex + programCount] = float_to_half(f2);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 65535;
    RET[programIndex + programCount] = 32768;
}
