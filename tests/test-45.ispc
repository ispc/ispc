#include "../test_static.isph"
task void f_v(uniform float RET[]) { float x = 10; RET[programIndex] = x--; }

task void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
