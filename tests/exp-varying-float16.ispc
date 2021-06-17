// rule: skip on arch=x86
// rule: skip on arch=x86-64
// rule: skip on cpu=TGLLP
bool ok(float16 x, float16 ref) { return (abs(x - ref) < 1d - 4) || abs((x - ref) / ref) < 1d - 3; }
export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying float16 arg = aFOO[programIndex] - (programCount / 2);
    varying float16 ref = exp((float)arg);
    varying float16 res = exp((float16)arg);
    RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

export void result(uniform float RET[]) { RET[programIndex] = 0.0; }
