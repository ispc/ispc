#include "test_static.isph"
// Negated unsuffixed integer literals must keep their value on targets with
// 8-bit and 16-bit data types, where small literals get narrow types.
task void f_v(uniform float RET[]) {
    uniform int a = -128;
    uniform int b = -130;
    uniform int c = -255;
    uniform int d = -32768;
    int e = -200 + programIndex;
    RET[programIndex] = (float)(a + b + c + d) + (float)e;
}

task void result(uniform float RET[]) { RET[programIndex] = -33481.0f + (float)programIndex; }
