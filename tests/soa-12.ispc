#include "../test_static.isph"
struct Point { float x, y, z; };


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    #pragma ignore warning(perf)
    soa<4> Point pts[2] = { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } },
                            { { 13, 14, 15, 16 }, { 17, 18, 19, 20, }, { 21, 22, 23, 24 } } };

    #pragma ignore warning(perf)
    RET[programIndex] = pts[programIndex & 1].y;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 6 : 5;
}
