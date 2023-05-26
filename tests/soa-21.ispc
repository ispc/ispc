#include "../test_static.isph"
// rule: skip on arch=xe64
struct Point { float x, y, z; };

struct Foo {
    int x;
    Point pts[10];
    int8 z;
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Foo * uniform pts = uniform new soa<8> Foo[11];
    foreach (i = 0 ... 88) {
        pts[i].x = b*i;
        pts[i].z = -b*i;
        for (uniform int j = 0; j < 10; ++j) {
            pts[i].pts[j].x = j + 100*i;
            pts[i].pts[j].y = j + 1000*i;
            pts[i].pts[j].z = j + 10000*i;
        }
    }
    
    assert(7+programCount < 88);
    soa<8> Foo * ptr = &pts[7+programIndex];
    #pragma ignore warning(perf)
    RET[programIndex] = ptr->pts[3].z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10000 * (7 + programIndex) + 3;
}
