#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float x = aFOO[programCount/2];
    uniform float d, invsqrt = rsqrt(x);
    d = (x * (invsqrt * invsqrt)) - 1.;
    if (d < 0.) d = -d;
    RET[programIndex] = (d > 1e-5) ? 1 : 0;
}


task void result(uniform float RET[]) {
       RET[programIndex] = 0;
}
