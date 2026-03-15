#include "test_static.isph"
// rule: run on OS=!windows
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
// rule: skip on target=generic.*
// TODO: LLVM 19 PPC64 backend cannot select v2f64 froundeven; fixed in later LLVM versions.
// rule: skip on arch=ppc64le

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex]; 
    RET[programIndex] = round(a+.49999); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + (double) programIndex;
}
