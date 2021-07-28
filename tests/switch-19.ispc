#include "../test_static.isph"
int switchit(int a) {
    int res = 0;
    for (int i = 0; i < programIndex; ++i) {
        switch(a) {
        case 1:
        case 2:
        case 3:
        case 4:
            res += (a - 1);
            break;
        default:
            res += a;
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            res += (a + 1);
            break;
        case 9:
        case 10:
        case 11:
        case 12:
            res += (a * 2);
        }
    }
    return res;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];
    int res = switchit(a);
    RET[programIndex] = res;
}

task void result(uniform float RET[]) {
    if (programIndex < 4)
        RET[programIndex] = programIndex * programIndex;
    else if (programIndex < 8)
        RET[programIndex] = programIndex * (programIndex + 2);
    else if (programIndex < 12)
        RET[programIndex] = programIndex * (programIndex + 1) * 2;
    else
        RET[programIndex] = programIndex * (programIndex + 1);
}
