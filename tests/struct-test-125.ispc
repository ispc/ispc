#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    varying struct { float<2> v; } a;
    a.v[0] = 1;
    a.v[1] = 2;
    RET[programIndex] = a.v[0] + a.v[1];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
