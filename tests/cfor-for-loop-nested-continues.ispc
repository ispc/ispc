#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int i, j;
    float r = 0;
    cfor (i = 0; i < a+1; ++i) {
        if (i == 1)
            continue;
        cfor (j = 0; j < a; ++j) {
            if (a == 2)
                continue;
        }
        ++r;
    }
    RET[programIndex] = r;
}

task void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
