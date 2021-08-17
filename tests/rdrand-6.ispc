#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
#ifndef ISPC_HAS_RAND
    RET[programIndex] = 0;
#else

    int set[32] = { 0 };
    uniform int count = 1024*1024;
    for (uniform int i = 0; i < count; ++i) {
        uniform int32 rr[programCount];
        int * ptr = rr + programIndex;
        while (!rdrand(ptr))
            ;
        int32 r = rr[programIndex];
        for (uniform int b = 0; b < 32; ++b)
            if (((unsigned int32)r >> b) & 1)
                ++set[b];
    }

    RET[programIndex] = 0;
    for (uniform int b = 0; b < 32; ++b)  {
        float r = (double)set[b] / (double)(count);
        if (!(r >= .49 && r < .51)) {
            ++RET[programIndex];
        }
    }
#endif
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
