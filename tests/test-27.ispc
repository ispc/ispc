#include "../test_static.isph"
task void f_v(uniform float RET[]) { if (true) { RET[programIndex] = (5%3) * (4./8.) + 2.14159; } else RET[programIndex] = 1.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
