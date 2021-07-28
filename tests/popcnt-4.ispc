#include "../test_static.isph"
task void f_f(uniform float RET[4], uniform float aFOO[]) {
    int64 a = aFOO[programIndex];
    a = (a < 3) ? 1 : 0;
    if ((programIndex & 1) != 0)
        a |= ((int64)1 << 36);
    RET[programIndex] = popcnt(a);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    if (programIndex & 1)
        ++RET[programIndex];
    if (programIndex < 2)
        ++RET[programIndex];
}
