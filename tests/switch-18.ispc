#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];

    int res = 0;
    RET[programIndex] = res;
    if (programIndex > 4) {
        switch (a) {
        case 6:
            res = 1;
            break;
        case 7:
        case 8:
        default:
            foreach_active(i) { res = a - i + 1; }
            break;
        case 9:
        case 10:
        case 11:
            res = (int)all(a >= 9 && a <= 11) + 2;
        }
    }
    RET[programIndex] = res;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
    RET[0] = RET[1] = RET[2] = RET[3] = RET[4] = 0;
    RET[5] = 1;
    RET[8] = RET[9] = RET[10] = 3;
}
