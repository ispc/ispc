#include "../test_static.isph"
int switchit(int a, uniform int b) {
    switch (a) {
    case 3:
        return 1;
    case 7:
        if (b == 5)
            break;
    default:
        return 0;
    }
    return -1;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[2] = 1;
    RET[6] = -1;
}
