#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];
    for (uniform int i = 0; i < 8*10; ++i) {
        pts[i].x = b*i;
        pts[i].y = 2*b*i;
        pts[i].z = 3*b*i;
    }

    assert(programCount < 80);
    RET[programIndex] = pts[programIndex].y;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10*programIndex;
}
