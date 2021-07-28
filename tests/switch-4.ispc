#include "../test_static.isph"
int switchit(int a, uniform int b) {
    int r = 0;
    switch (a) {
    case 3:
        r = 1;
        break;
    default:
        r = 0;
    }
    return r;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex == 2) ? 1 : 0;
}
