#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = popcnt((int)a);
}

static int manualpc(int v) {
    int count = 0;
    while (v != 0) {
        if (v & 1)
            ++count;
        v >>= 1;
    }
    return count;
}

task void result(uniform float RET[]) {
    assert(programCount <= 64);
    RET[programIndex] = manualpc(programIndex+1);
}
