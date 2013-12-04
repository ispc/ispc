
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int16 a = 65535, b = 65535; // max unsigned int16
    RET[programIndex] = paddus(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 65535;
}
