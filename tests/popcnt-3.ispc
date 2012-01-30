
export uniform int width() { return programCount; }



export void f_f(uniform float RET[4], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = popcnt(a < 3);
}

export void result(uniform float RET[]) { RET[programIndex] = programCount == 1 ? 1 : 2; }
