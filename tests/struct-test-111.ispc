#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    varying struct { float x, y; int i; } a;
    a.x = 1;
    a.y = 2;
    RET[programIndex] = a.x + a.y;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
