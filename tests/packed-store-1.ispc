#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int pack[2+programCount];
    for (uniform int i = 0; i < 2+programCount; ++i)
        pack[i] = 0;
    packed_store_active(&pack[2], a);
    RET[programIndex] = pack[programIndex]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex-1;
    RET[0] = RET[1] = 0;
}
