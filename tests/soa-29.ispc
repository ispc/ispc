#include "../test_static.isph"
struct Point { float x, y; int8 zzz0; float z; double aa[3]; int8 zzz; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[20];

    for (int i = programIndex; i < 32*b; i += programCount) {
        pts[i].x = b*i;
        pts[i].y = 2*b*i;
        pts[i].z = 3*b*i;
    }

    uniform Point up = pts[4];

    assert(2*programIndex < 160);
    #pragma ignore warning(perf)
    RET[programIndex] = pts[2*programIndex].x;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10 * programIndex;
}
