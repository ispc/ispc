#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float16 x = aFOO[programCount/2];
    uniform float16 d, invsqrt = rsqrt(x);
    d = (x * (invsqrt * invsqrt)) - 1.0f16;
    if (d < 0.0f16) d = -d;
    RET[programIndex] = (d > 1e-2) ? 1 : 0;
}

task void result(uniform float RET[]) {
       RET[programIndex] = 0;
}
