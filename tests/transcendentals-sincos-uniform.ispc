// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform float ref;
    uniform float ref2;
    sincos((float)arg, &ref, &ref2);
    uniform double res;
    uniform double res2;
    sincos(arg, &res, &res2);
    uniform double delta = abs((res - ref) / ref);
    uniform double delta2 = abs((res2 - ref2) / ref2);
    // native single precision sin and cos on genx is pretty imprecise
    RET[i] = (delta < 1d-4) && (delta2 < 1d-4) ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
