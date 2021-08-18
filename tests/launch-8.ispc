#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64

#define N0 12
#define N1 20
#define N2 50
static uniform float array[N2][N1][N0];

task void x(const uniform float farray[]) {
    const float f = farray[programIndex];
    uniform int j;

    assert(taskCount  == (uniform int32)N0*N1*N2);
    assert(taskCount0 == (uniform int32)N0);
    assert(taskCount1 == (uniform int32)N1);
    assert(taskCount2 == (uniform int32)N2);
    assert(taskIndex  == (uniform int32)taskIndex0 + (uniform int32)N0*(taskIndex1 +(uniform int32) N1*taskIndex2));
    assert(taskIndex0 < (uniform int32)N0);
    assert(taskIndex1 < (uniform int32)N1);
    assert(taskIndex2 < (uniform int32)N2);

    const uniform int i0 = taskIndex0;
    const uniform int i1 = taskIndex1;
    const uniform int i2 = taskIndex2;
    const uniform int i  = taskIndex;
    array[i2][i1][i0] = i / 10000.;
    #pragma ignore warning
    cfor (j = 0; j < 10000; ++j)
        array[i2][i1][i0] = sin(array[i2][i1][i0]);
    if (array[i2][i1][i0] < .02)
        array[i2][i1][i0] = i;
}
task void f_f(uniform float RET[], uniform float fFOO[]) {
    uniform float * uniform f = fFOO;
    launch[N2][N1][N0] x(f);
    sync;
    RET[programIndex] = array[N2-1][N1-1][N0-1];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 11999.000000;
}
