#include "../test_static.isph"
// rule: skip on arch=xe64
struct Point { float x, y, z; };

struct Foo {
    float<3> vec;
    int8 z;
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Foo * uniform pts = uniform new soa<8> Foo[10];
    foreach (i = 0 ... 80) {
        pts[i].vec.x = b*i;
        pts[i].vec.y = -b*i;
        pts[i].vec.z = 2*b*i;
        pts[i].z = i;
    }

    assert(programIndex + 2 < 80);
    #pragma ignore warning(perf)
    RET[programIndex] = pts[programIndex+2].vec.y;
}

task void result(uniform float RET[]) {
    RET[programIndex] = -(5 * (programIndex+2));
}
