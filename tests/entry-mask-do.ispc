#include "../test_static.isph"
int * uniform ptr = NULL;

int func(int v) {
    int ret;
    //    print("%\n", v);
    do {
        if (v == 0) {
            ret = 1;
            break;
        }
        *ptr = 1;
    } while(true);
    return ret;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int count = 10;
    if (programIndex & 1)
        count = 0x7ffffff;
    RET[programIndex] = 0;
    if (!(programIndex & 1))
        RET[programIndex] = func(programIndex & 1);
}

task void result(uniform float RET[]) {
    RET[programIndex] = !(programIndex & 1);
}
