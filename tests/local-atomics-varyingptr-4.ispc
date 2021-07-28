#include "../test_static.isph"
uniform int32 s[programCount];

task void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        s[i] = -1234;
    atomic_max_local(&s[programIndex], programIndex);
    RET[programIndex] = s[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
