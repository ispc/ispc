#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    varying struct { float<2> v; } a;
    a.v.x = 1;
    a.v.y = 2;
    RET[programIndex] = a.v.x + a.v.y;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
