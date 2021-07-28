#include "../test_static.isph"
uniform int foo(int i);

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float ub) {
    float b = ub;
    RET[programIndex] = 0;
    
    foreach (i = 0 ... 10, j = 0 ... programCount + 1) {
        if (ub == 12345)
            continue;
        if (j > 0)
            ++RET[j-1];
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
