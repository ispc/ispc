#include "../test_static.isph"
// rule: skip on arch=xe64
typedef task void (*TaskFn)(float i);

struct Test {
    TaskFn fn;
};
static uniform float array[10000];

task void x(float f) {
    uniform int j;
    uniform int i = taskIndex;
    array[i] = i / 10000.;
    #pragma ignore warning
    cfor (j = 0; j < 10000; ++j)
        array[i] = sin(array[i]);
    if (array[i] < .02)
        array[i] = i;
}
task void f_f(uniform float RET[], uniform float fFOO[]) {
    float f = fFOO[programIndex];
    uniform TaskFn func = x;
    launch [10000] func(f);
    sync;
    RET[programIndex] = array[9999];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 9999.000000;
}
