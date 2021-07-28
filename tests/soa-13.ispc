#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];

    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].y = 2*b*i;
        pts[i].z = 3*b*i;
    }

    uniform Point up = { b, 3, 170 };
    pts[(int64)1] = up;

    assert(programCount < 80);
    RET[programIndex] = pts[(int64)programIndex].z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 15*programIndex;
    RET[1] = 170;
}
