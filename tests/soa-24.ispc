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

    #pragma ignore warning(perf)
    pts[programIndex+2].vec.z *= -1;    
    assert(programIndex < 80);
    float<3> vl = pts[programIndex].vec;
    RET[programIndex] = vl.z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10 * programIndex;
    if (programIndex >= 2)
        RET[programIndex] *= -1;
}
