#include "test_static.isph"
// rule: skip on arch=xe64

// A launched task gets its bool parameters through the argument block of the
// launch, which holds their register types (a varying bool is the mask).

static uniform int got[programCount];
static uniform int gotUniform;
static uniform int gotShort;

task void x(bool v, uniform bool u, uniform bool<3> s) {
    got[programIndex] = v ? 1 : 0;
    gotUniform = u ? 1 : 0;
    gotShort = (s[0] ? 1 : 0) + (s[1] ? 2 : 0) + (s[2] ? 4 : 0);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform bool<3> s = {true, false, true};
    launch[1] x(programIndex % 2 == 1, true, s);
    sync;
    RET[programIndex] = got[programIndex] + 10 * gotUniform + 100 * gotShort;
}

task void result(uniform float RET[]) { RET[programIndex] = (programIndex % 2) + 10 + 500; }
