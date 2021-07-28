#include "../test_static.isph"
task void f_v(uniform float RET[]) { float x = 1.;  x = 2; RET[programIndex] = x; }

task void result(uniform float RET[]) {
    RET[programIndex] = 2.000000;
}
