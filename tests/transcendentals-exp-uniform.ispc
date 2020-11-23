// rule: skip on cpu=TGLLP
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-7) || abs((x-ref)/ref) < 1d-6; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double ref = exp((float)aFOO[i]);
    RET[i] = ok(exp(aFOO[i]), ref) ? 0. : 1.;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
