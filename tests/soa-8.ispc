#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[12];
    foreach (i = 0 ... 96) {
        pts[i].x = -42;
        pts[i].y = 0;
        pts[i].z = 0;
    }

    Point pv = { a, b, -a };
    assert(8+programCount < 96);
    #pragma ignore warning(perf)
    pts[7+programIndex] = pv;

    RET[programIndex] = pts[8+programIndex].x;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (2 + programIndex);
    RET[programCount-1] = -42;
}
