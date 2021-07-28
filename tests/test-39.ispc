#include "../test_static.isph"
task void f_v(uniform float RET[]) { RET[programIndex] = (2 < 3) ? 42. : -1.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 42.000000;
}
