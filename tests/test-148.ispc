#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double x = aFOO[programIndex]*1d100;
    double d, invsqrt = rsqrt(x);
    d = (x * (invsqrt * invsqrt)) - 1.0d0;
    if (d < 0.0d0) d = -d;
    RET[programIndex] = d > 1d-15;
}


task void result(uniform float RET[]) {
       RET[programIndex] = 0;
}
