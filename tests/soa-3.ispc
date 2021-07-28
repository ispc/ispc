#include "../test_static.isph"
struct Point { float x, y[3], z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];

    soa<8> Point pts[10];
//CO    uniform Point pts[80];
    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].y[0] = 2*b*i;
        pts[i].y[1] = 2*b*i+1;
        pts[i].y[2] = 2*b*i+2;
        pts[i].z = 3*b*i;
    }

    assert(programCount < 80);
    RET[programIndex] = pts[programIndex].y[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10*programIndex + 2;
}
