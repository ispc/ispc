#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
bool ok(double x, double ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-4; }

task void f_v(uniform float RET[]) {
    uniform double vals_d[8] = { 0, 1, 0.5, -1, -.87, -.25, 1e-3, -.99999999 };
    uniform double r_d[8];
    foreach (i = 0 ... 8) {
        // TODO: Possible optimization opportunity.
        #pragma ignore warning(perf)
        r_d[i] = sin(asin(vals_d[i%8]));    //testing varying double
    }
    int errors = 0;
    for (uniform int i = 0; i < 8; ++i) {
        if (ok(r_d[i], vals_d[i%8]) == false) {
#ifndef ISPC_GPU
            print("error @ %: got %, expected %\n", i, r_d[i], vals_d[i%8]);
#endif
            ++errors;
        }
    }
    for (uniform int i = 0; i < 8 ; i++){
        r_d[i] = sin(asin(vals_d[i%8]));  //testing uniform double
    }
    for (uniform int i = 0; i < 8; ++i) {
        if (ok(r_d[i], vals_d[i%8]) == false) {
#ifndef ISPC_GPU
            print("error @ %: got %, expected %\n", i, r_d[i], vals_d[i%8]);
#endif
            ++errors;
        }
    }
    RET[programIndex] = errors;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
