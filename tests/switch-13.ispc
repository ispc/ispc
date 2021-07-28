#include "../test_static.isph"
int switchit(int a, uniform int b) {
    int r = -1;
    switch (b) {
    case 5:
        if (a & 1) {
            r=3;
            break;
        }
        r= 2;
        break;
    default:
        r= 3;
    }
    return r;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 2 : 3;
}
