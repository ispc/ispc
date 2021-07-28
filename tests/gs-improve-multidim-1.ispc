#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[2][programCount+5];
    for (uniform int i = 0; i < 2; ++i)
        for (uniform int j = 0; j < programCount+5; ++j)
            udx[i][j] = 100*i+j;

    int x = 1;
    RET[programIndex] = udx[x][programIndex+1];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 101+programIndex;
}
