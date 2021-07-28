#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    varying struct { varying struct { float x, y; } q; int i; } a[3];
    a[1].q.x = 1;
    a[1].q.y = 2;
    RET[programIndex] = a[1].q.x + a[1].q.y;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
