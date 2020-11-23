// rule: skip on cpu=TGLLP
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-6) || abs((x-ref)/ref) < 1d-5; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i] / (programCount + 2.0);
    uniform double ref = asin((float)arg);
    uniform double res = asin(arg);
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
