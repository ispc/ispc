export void f_f(uniform float RET[], uniform float aFOO[]) {
  RET[programIndex] = sizeof 1u;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
