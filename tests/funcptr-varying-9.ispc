#include "../test_static.isph"
typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return a<b?a:b;
}


task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1; // == 0
    FuncType func[] = { (a == 1) ? foo : bar, (a == 1) ? bar : foo };
    #pragma ignore warning(perf)
    FuncType f = func[(a == 1) ? 0 : 1];
    RET[programIndex] = f(a,b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex+1;
}
