#include "../test_static.isph"
task void f_v(uniform float RET[]) { if (1. && true) RET[programIndex] = 10.; else RET[programIndex] = -1.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
