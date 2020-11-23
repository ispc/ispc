// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
bool ok(double x, double ref) { return (abs(x - ref) < 1d-5) || abs((x-ref)/ref) < 1d-4; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex] + b;
  varying double ref = cos((float)arg);
  varying double res = cos(arg);
  // native single precision cos on genx is pretty imprecise
  RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
