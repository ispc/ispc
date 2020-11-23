// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-6) || abs((x-ref)/ref) < 1d-5; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform double arg2 = aFOO[i] / b;
    uniform double ref = pow((float)arg, (float)arg2);
    uniform double res = pow(arg, arg2);
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
