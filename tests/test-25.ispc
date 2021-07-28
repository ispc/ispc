#include "../test_static.isph"
task void f_v(uniform float RET[]) { RET[programIndex] = 3. * 4 - 7 + (0 << 4); }

task void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
