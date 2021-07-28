#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];

    int res = 0;
    if (programIndex & 1) {
        switch (a) {
            case 6:
                res = 1;
                break;
            default:
                res = 3;
                break;
        }
    }
    RET[programIndex] = res;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? (programIndex == 5 ? 1 : 3) : 0;
}
