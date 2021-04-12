// sincos() function is available only on Linux.
// rule: skip on OS=*
// rule: run on OS=linux
// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
bool ok(double x, double ref) { return (abs(x - ref) < 1d-5) || abs((x-ref)/ref) < 1d-4; }
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex];
  varying float ref;
  varying float ref2;
  sincos((float)arg, &ref, &ref2);
  varying double res;
  varying double res2;
  sincos(arg, &res, &res2);
  RET[programIndex] = ok(res, ref) && ok(res2, ref2) ? 0.0 : 1.0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
