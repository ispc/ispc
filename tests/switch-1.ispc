#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    switch (b) {
    default:
        RET[programIndex] = -1; 
        break;
    case 5:
        RET[programIndex] = 0; 
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
