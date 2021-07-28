#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[11];
    foreach (i = 0 ... 88) {
        pts[i].x = -42;
        pts[i].y = 0;
        pts[i].z = 0;
    }

    Point pv = { a, b, -a };
    assert(8+programCount < 88);
    pts[8+programIndex] = pv;

    #pragma ignore warning(perf)
    RET[programIndex] = pts[6+programIndex].x;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (1 + programIndex - 2);
    RET[0] = RET[1] = -42;
}
