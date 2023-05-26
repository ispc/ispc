#include "../test_static.isph"
// rule: skip on arch=xe64

static uniform float array[10];
task void foo(uniform float f) { array[0] = f; }
task void foo(uniform float f, uniform int i) { array[i] = f; }
task void f_v(uniform float RET[]) {
    launch foo(12.);
    launch foo(-1., 1);
    sync;
    RET[programIndex] = array[0] + array[1];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 11.000000;
}
