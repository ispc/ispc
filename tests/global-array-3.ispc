#include "../test_static.isph"
float a[100];



task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = i*programCount+programIndex;
    RET[programIndex] = a[b-5]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
