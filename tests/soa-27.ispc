#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];

    foreach (i = 1 ... 80) {
        #pragma ignore warning(perf)
        pts[i].x = b*i;
        #pragma ignore warning(perf)
        pts[i].y = 2*b*i;
        #pragma ignore warning(perf)
        pts[i].z = 3*b*i;
    }
    pts[0].x = pts[0].y = pts[0].z = 0;

    uniform Point up = pts[4];

    RET[programIndex] = up.z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 60;
}
