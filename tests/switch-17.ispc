#include "../test_static.isph"
int func(int a) {
    int res = 0;
    switch (a) {
        case 0:
            res = 1;
            break;
        default:
            res = 2;
            break;
    }
    return res;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];

    int res = 0;
    if (programIndex & 1) {
        res = func(a & 1);
    }
    RET[programIndex] = res;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex & 1 ? 1 : 0;
}
