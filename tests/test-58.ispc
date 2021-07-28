#include "../test_static.isph"
float bar(float x);
task void f_v(uniform float RET[]) { RET[programIndex] = bar(3.); }
float bar(float x) { return 2. * x; }

task void result(uniform float RET[]) {
    RET[programIndex] = 6.000000;
}
