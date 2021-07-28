#include "../test_static.isph"
task void f_v(uniform float RET[]) { int x = 10.;  x = 2.; RET[programIndex] = (float)x; }

task void result(uniform float RET[]) {
    RET[programIndex] = 2.000000;
}
