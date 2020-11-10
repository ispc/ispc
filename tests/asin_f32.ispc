bool ok(float x, float ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-4; }

export void f_v(uniform float RET[]) {
    uniform float vals_f[8] = { 0, 1, 0.5, -1, -.87, -.25, 1e-3, -.99999999 };
    uniform float r_f[8];
    foreach (i = 0 ... 8) {  //testing varying float
        // TODO: Possible optimization opportunity.
        #pragma ignore warning(perf)
        r_f[i] = sin(asin(vals_f[i%8]));
    }

    int errors = 0;
    for (uniform int i = 0; i < 8; ++i) {
        if (ok(r_f[i], vals_f[i%8]) == false) {
#ifndef ISPC_GENX_ENABLED
            print("error @ %: got %, expected %\n", i, r_f[i], vals_f[i%8]);
#endif
            ++errors;
        }
    }
    for (uniform int i = 0; i < 8 ; i++){
        r_f[i] = sin(asin(vals_f[i%8]));  //testing uniform float
    }
    for (uniform int i = 0; i < 8; ++i) {
        if (ok(r_f[i], vals_f[i%8]) == false) {
#ifndef ISPC_GENX_ENABLED
            print("error @ %: got %, expected %\n", i, r_f[i], vals_f[i%8]);
#endif
            ++errors;
        }
    }
    RET[programIndex] = errors;
}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
