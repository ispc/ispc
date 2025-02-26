#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform unsigned int16 pack[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        pack[i] = 0;
    #pragma ignore warning(perf)
    packed_store_active2(pack, (unsigned int16)a);
    #pragma ignore warning(perf)
    RET[programIndex] = pack[programIndex]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
