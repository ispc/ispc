static float x = 1;

export void f_v(uniform float RET[]) { RET[programIndex] = x; }

export void result(uniform float RET[]) { RET[programIndex] = 1.; }
