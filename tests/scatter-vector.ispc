#include "../test_static.isph"
struct Foo {
    float f;
    uniform int a;
    float y;
};

typedef float<3> float3;

void set(uniform float3 f[], int offset, float3 val) {
    #pragma ignore warning(perf)
    f[offset] = val;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 

    uniform float3 foo[programCount+5];
    for (uniform int i = 0; i < programCount+5; ++i) {
        foo[i].x = i;
        foo[i].y = -1;
        foo[i].z = 2*i;
    }

    float3 bar = { programIndex, 2*programIndex, 3*programIndex };
    set(foo, a-1, bar);

    #pragma ignore warning(perf)
    RET[programIndex] = foo[programIndex].z; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3*programIndex;
}
