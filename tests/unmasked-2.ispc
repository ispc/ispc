#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    int count = 0;
    for (int i = 0; i < a; ++i) {
        unmasked {
            ++count;
        }
    }
    RET[programIndex] = count; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount;
}
