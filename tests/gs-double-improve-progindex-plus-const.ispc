#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[17] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 };
    int x = (programIndex + 1) % 17;
    RET[programIndex] = udx[x];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+((1+programIndex)%17); 
}
