
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int16 a = 32767, b = 32767; // max signed int16
    RET[programIndex] = padds(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 32767;
}
