
export uniform int width() { return programCount; }



typedef float Float;

export void f_fu(uniform float RET[], uniform Float aFOO[], uniform Float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = a+b;
}

export void result(uniform float RET[]) { RET[programIndex] = 6 + programIndex; }
