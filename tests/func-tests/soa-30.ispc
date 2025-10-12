#include "test_static.isph"
struct Point { float x, y, z; };

// Similar to soa-1.ispc, with soa<> wrapped in struct
struct SoaPoints {
    soa<8> Point pts[10];
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];

    SoaPoints s;
    for (uniform int i = 0; i < 8*10; ++i) {
        s.pts[i].x = b*i;
        s.pts[i].y = 2*b*i;
        s.pts[i].z = 3*b*i;
    }

    assert(programCount < 80);
    RET[programIndex] = s.pts[programIndex].y;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10*programIndex;
}
