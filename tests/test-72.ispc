#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    int i; float r = 0.;
    for (i = 0; i < 10; ++i) { r += 2; }
    RET[programIndex] = r;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 20.000000;
}
