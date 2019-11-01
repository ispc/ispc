export void f_v(uniform float RET[]) {
int i=0, j=0;
cwhile (i++ < 10) {
  ++j;
  if (i >= 5) continue;
  ++j;}
RET[programIndex] = (float)j; }

export void result(uniform float RET[]) {
    RET[programIndex] = 14.000000;
}
