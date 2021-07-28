#include "../test_static.isph"
// make sure we reset the func mask in the foreach loop...

void update(uniform float val[], const uniform float a[]) {
    foreach (i = 0 ... programCount+3) {
        int ic = min(i, programCount-1);
        #pragma ignore warning(perf)
        val[i] += a[ic] - 1 + i-ic;
    }
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount+3];
    for (uniform int i = 0; i < programCount+3; ++i)
        val[i] = 0;

    if ((int)aFOO[programIndex] & 1)
        update(val, aFOO);

    RET[programIndex] = val[3+programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex+3;
}
