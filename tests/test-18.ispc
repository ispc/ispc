#include "../test_static.isph"
task void f_v(uniform float RET[]) { RET[programIndex] = (-1. + 2) -  1; }

task void result(uniform float RET[]) {
    RET[programIndex] = 0.000000;
}
