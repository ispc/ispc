#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    if (a < 0)
        // Though this condition is false, the assert shouldn't hit, since
        // given the if test above, all lanes should be off.
        assert(b == 0.);
    RET[programIndex] = a; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
