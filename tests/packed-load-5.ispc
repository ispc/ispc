#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[100];
    for (uniform int i = 0; i < 100; ++i)
        a[i] = i;
    int aa = 32;
    uniform int count = 0;
    if (programIndex & 1)
        count += packed_load_active(&a[10], &aa);
    if (!(programIndex & 1))
        count += packed_load_active(&a[10+count], &aa);
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) {
    if (programCount == 1)
        RET[0] = 10;
    else {
        for (uniform int i = 0; i < programCount/2; ++i) {
            RET[2*i+1] = 10+i;
            RET[2*i] = 10+programCount/2+i;
        }
    }
}
