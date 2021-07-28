#include "../test_static.isph"
task void f_v(uniform float RET[]) { float z=-3.; RET[programIndex] = z+3.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 0.000000;
}
