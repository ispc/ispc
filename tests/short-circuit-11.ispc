#include "../test_static.isph"


uniform int * uniform ptr;

bool crashEven() {
//CO    return (programIndex & 1) ? true : (*ptr > 0);
    if (programIndex & 1) 
        return true;
    else
        return (*ptr > 0);
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float a0 = aFOO[0], a1 = aFOO[1];
    if (((programIndex & 1) == 1) && crashEven())
        RET[programIndex] = 1;
    else
        RET[programIndex] = 2;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 1 : 2;
}
