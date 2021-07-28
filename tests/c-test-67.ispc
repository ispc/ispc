#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, j;
    for (i = 0; i < b; ++i) {
        if (a == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = programIndex < 5 ? 5 : 6+programIndex;
}
