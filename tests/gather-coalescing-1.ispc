#include "../test_static.isph"
#define LOAD(n)               \
    do {                      \
        if (n < programCount) \
            a[n] = aFOO[n];   \
    } while (false)

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[programCount];

    // Initialization
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = 0;

    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);
    LOAD(6);
    LOAD(7);
    LOAD(8);
    LOAD(9);
    LOAD(10);
    LOAD(11);
    LOAD(12);
    LOAD(13);
    LOAD(14);
    LOAD(15);

    uniform int elem_num = programCount > 16 ? 16 : programCount;

    // Perform calculation on loaded values
    for (uniform int i = 0; i < elem_num; ++i)
        a[i] *= (i + 1);

    RET[programIndex] = a[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex + 1) * (programIndex + 1);

    // Not loaded lanes
    for (uniform int i = 16; i < programCount; ++i)
        RET[i] = 0;
}
