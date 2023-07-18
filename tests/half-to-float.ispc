#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    unsigned int16 h = 65535;
    float f = half_to_float(h);
    RET[programIndex] = float_to_half(f);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 65535;
}
