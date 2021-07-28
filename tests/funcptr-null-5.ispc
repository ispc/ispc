#include "../test_static.isph"
typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return min(a, b);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1;
    FuncType func = NULL;
    if (a == 2)
        func = foo;
    if (func)
        RET[programIndex] = func(a, b);
    else
        RET[programIndex] = -1;
}

task void result(uniform float RET[]) {
    RET[programIndex] = -1;
    RET[1] = 2;
}
