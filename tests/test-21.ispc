#include "../test_static.isph"
task void f_v(uniform float RET[]) { RET[programIndex] = (1<<1) + 1.14159; }

task void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
