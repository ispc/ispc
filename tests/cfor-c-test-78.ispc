
export uniform int width() { return programCount; }


export void f_v(uniform float RET[]) {
int i;
cfor (i = 0; i < 10; ++i) {
  if (i == 5) cbreak;
}
RET[programIndex] = (float)i; }

export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
