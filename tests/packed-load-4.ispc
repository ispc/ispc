#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[100];
    for (uniform int i = 0; i < 100; ++i)
        a[i] = i;
    int aa = 32;
    uniform int count = 0;
    if (programIndex < 2)
        count += packed_load_active(&a[5], &aa);
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 32;
    RET[0] = 5;
    RET[1] = 6;
}
