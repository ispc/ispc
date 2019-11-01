export void f_v(uniform float RET[]) {
int i, j=0;
cfor (i = 0; i < 10; ++i) {
  if (i >= 5) continue;
  ++j;}
RET[programIndex] = (float)j; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
