#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    const uniform float x = 1.;
    RET[programIndex] = x+x;
}


task void result(uniform float RET[]) { RET[programIndex] = 2; }
