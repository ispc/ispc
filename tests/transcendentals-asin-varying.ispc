// rule: skip on cpu=TGLLP
bool ok(double x, double ref) { return (abs(x - ref) < 1d-6) || abs((x-ref)/ref) < 1d-5; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex] / (programCount + 2.0);
  varying double ref = asin((float)arg);
  varying double res = asin(arg);
  RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
