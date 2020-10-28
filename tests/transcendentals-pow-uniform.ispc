// rule: skip on arch=genx32
// rule: skip on arch=genx64
// rule: skip on cpu=TGLLP
// currently cos/sin/pow/sincos is very slow on gen so skip it
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform double arg2 = aFOO[i] / b;
    uniform double ref = pow((float)arg, (float)arg2);
    uniform double res = pow(arg, arg2);
    uniform double delta = abs((res - ref) / ref);
    RET[i] = delta < 1d-6 ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
