#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    bool a;
    for (uniform int i = 0; i < programCount; ++i) {
        uniform bool insertVal = ((i % 3) == 2);
        a = insert(a, i, insertVal);
    }
    if (a)
        RET[programIndex] = 6;
    else
        RET[programIndex] = 7;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 7;
    if ((programIndex % 3) == 2)
        RET[programIndex] = 6;
}
