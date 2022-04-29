#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float16 x = aFOO[programIndex/2];
    varying float16 d, invsqrt = rsqrt(x);
    d = (x * (invsqrt * invsqrt)) - 1.0f16;
    // WA: print changes the behavior of the test
    // print("WA: %\n", d);
    if (d < 0.0f16) d = -d;
    if (d < 1e-2) {
        RET[programIndex] = 1;
    }
    else {
        RET[programIndex] = 0;
    }
}


task void result(uniform float RET[]) {
       RET[programIndex] = 1;
}
