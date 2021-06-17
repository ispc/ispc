// rule: skip on arch=x86
// rule: skip on arch=x86-64
// rule: skip on cpu=TGLLP
uniform bool ok(uniform float16 x, uniform float16 ref) {
    return (abs(x - ref) < 1d - 4) || abs((x - ref) / ref) < 1d - 3;
}
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    for (uniform int i = 0; i != programCount; ++i) {
        uniform float16 ref = exp((uniform float16)aFOO[i]);
        RET[i] = ok(exp(aFOO[i]), ref) ? 0. : 1.;
    }
}

export void result(uniform float RET[]) { RET[programIndex] = 0.0; }
