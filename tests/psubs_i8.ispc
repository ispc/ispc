
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int8 a = -128, b = 127; // min and max signed int8
    RET[programIndex] = psubs(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = -128;
}
