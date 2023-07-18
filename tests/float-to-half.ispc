#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    float f = 65535;
    unsigned int16 h = float_to_half(f);
    RET[programIndex] = half_to_float(h);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 65535;
}
