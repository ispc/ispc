
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int16 a = -32768, b = 32767; // min and max unsigned int16
    RET[programIndex] = psubs(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = -32768;
}
