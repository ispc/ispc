#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float x_fullFOO[], uniform float a) {
    float x_full = x_fullFOO[programIndex];
    static const float two_over_pi_vec = 2;
    float scaled = x_full * two_over_pi_vec;
    RET[programIndex] = scaled;
}

task void result(uniform float RET[]) { RET[programIndex] = 2+2*programIndex; }
