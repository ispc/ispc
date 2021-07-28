#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, j;
    for (i = 0; i < b; ++i) {
        if (a == b) 
            a += 10;
        ++a;
        if (a == 2) break;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 16+programIndex;
    if (programIndex >= 5) RET[programIndex] = 6+programIndex;
    RET[0] = 2;
}
