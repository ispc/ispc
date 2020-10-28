// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex] / 16.0;
  varying double ref = asin((float)arg);
  varying double res = asin(arg);
  varying double delta = abs((res - ref) / ref);
  RET[programIndex] = delta < 1d-5 ? 0.0 : 1.0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
