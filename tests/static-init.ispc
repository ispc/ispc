#include "../test_static.isph"
static float x = 1;

task void f_v(uniform float RET[]) { RET[programIndex] = x; }

task void result(uniform float RET[]) { RET[programIndex] = 1.; }
