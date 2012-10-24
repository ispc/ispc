
export uniform int width() { return programCount; }


export void f_v(uniform float RET[]) { int x = 1; x <<= 3; RET[programIndex] = (float)x; }

export void result(uniform float RET[]) {
    RET[programIndex] = 8.000000;
}
