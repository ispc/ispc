#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[5][5];
    for (uniform int i = 0; i < 5; ++i)
        for (uniform int j = 0; j < 5; ++j)
            udx[i][j] = 10*i+j;

    int x = 1;
    RET[programIndex] = udx[x][x];
}

task void result(uniform float RET[]) { RET[programIndex] = 11; }
