#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = a; 

    if (programIndex & 1) {
        foreach_active (i) {
            if (i & 1)
                continue;

            int * uniform null = 0;
            *null = 0;
        }
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = (1 + programIndex);
}
