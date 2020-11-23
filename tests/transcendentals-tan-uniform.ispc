// rule: skip on cpu=TGLLP
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-4) || abs((x-ref)/ref) < 1d-3; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i] + b;
    uniform double ref = tan((float)arg);
    uniform double res = tan(arg);
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
