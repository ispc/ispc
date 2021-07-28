#include "../test_static.isph"
struct Foo { float f; };

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float foo[5] = { 0, a, 0, 2 * a, 3 };
    uniform int offset[4] = { 1,2,3,4 };
    #pragma ignore warning(perf)
    int x = offset[programIndex & 0x3];
    #pragma ignore warning(perf)
    RET[programIndex] = foo[x];
}

task void result(uniform float RET[]) {
    for (uniform int i = 0; i < programCount; i += 4) {
        RET[i] = i+1;
        RET[i+1] = 0;
        RET[i+2] = 2 * (i+3);
        RET[i+3] = 3;
    }
}
