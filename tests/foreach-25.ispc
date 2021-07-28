#include "../test_static.isph"
uniform int foo(int i);

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int x[21];

    foreach (i = 0 ... 21) {
        x[i] = 0;
        if (x[i] != 12345)
            continue;
        x[i] = 1;
    }

    #pragma ignore warning(perf)
    RET[programIndex] = x[min(programIndex, 20)];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
