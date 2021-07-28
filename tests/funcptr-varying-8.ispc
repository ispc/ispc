#include "../test_static.isph"
typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return min(a, b);
}


task void f_f(uniform float RET[], uniform float aFOO[]) {
    FuncType func[] = { foo, bar };
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1; // == 0
    RET[programIndex] = func[(int)b](a, b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex+1;
}
