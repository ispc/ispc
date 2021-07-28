#include "../test_static.isph"
task void f_v(uniform float RET[]) { float x = 10, y; y = x++; RET[programIndex] = y; }

task void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
