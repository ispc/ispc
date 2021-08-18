#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64
struct Point { float x, y, z; };

struct Foo {
    int x;
    Point *pts[3];
    int8 z;
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Foo * uniform pts = uniform new soa<8> Foo[10];
    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].z = -b*i;
        for (uniform int j = 0; j < 3; ++j) {
            pts[i].pts[j] = new uniform Point[4];
            for (uniform int k = 0; k < 4; ++k) {
                #pragma ignore warning(perf)
                pts[i].pts[j][k].x = 100*i+10*j+k;
                #pragma ignore warning(perf)
                pts[i].pts[j][k].y = -1234;
                #pragma ignore warning(perf)
                pts[i].pts[j][k].z = -(100*i+10*j+k);
            }
        }
    }
   
    assert(programIndex < 80);
    #pragma ignore warning(perf)
    RET[programIndex] = pts[programIndex].pts[programIndex % 3][programIndex % 4].z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = -(100*programIndex+10*(programIndex % 3)+(programIndex % 4));
}
