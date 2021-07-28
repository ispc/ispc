#include "../test_static.isph"
task void f_v(uniform float RET[]) { if (~0 > 0) RET[programIndex] = 1.; RET[programIndex] = 2.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 2.000000;
}
