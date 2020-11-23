// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
bool ok(double x, double ref) { return (abs(x - ref) < 1d-6) || abs((x-ref)/ref) < 1d-5; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex];
  varying double arg2 = aFOO[programIndex] / b;
  varying double ref = pow((float)arg, (float)arg2);
  varying double res = pow(arg, arg2);
  RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
