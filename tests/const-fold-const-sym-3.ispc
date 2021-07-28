#include "../test_static.isph"
static const uniform float x = 1.;

task void f_v(uniform float RET[]) {
    RET[programIndex] = x+x;
}


task void result(uniform float RET[]) { RET[programIndex] = 2; }
