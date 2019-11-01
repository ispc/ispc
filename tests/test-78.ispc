export void f_v(uniform float RET[]) {
int i;
for (i = 0; i < 10; ++i) {
  if (i == 5) break;
}
RET[programIndex] = (float)i; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
