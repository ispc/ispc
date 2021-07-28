#include "../test_static.isph"
task void f_v(uniform float RET[]) { int x = 1; x <<= 3; RET[programIndex] = (float)x; }

task void result(uniform float RET[]) {
    RET[programIndex] = 8.000000;
}
