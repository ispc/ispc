// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex] + b;
  varying double ref = log((float)arg);
  varying double res = log(arg);
  varying double delta = abs((res - ref) / ref);
  RET[programIndex] = delta < 1d-7 ? 0.0 : 1.0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
