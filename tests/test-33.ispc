#include "../test_static.isph"
task void f_v(uniform float RET[]) { if (!true) RET[programIndex] = -1.; else RET[programIndex] = 10.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
