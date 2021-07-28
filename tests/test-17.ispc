#include "../test_static.isph"
task void f_v(uniform float RET[]) { RET[programIndex] = 12.5; }

task void result(uniform float RET[]) {
    RET[programIndex] = 12.500000;
}
