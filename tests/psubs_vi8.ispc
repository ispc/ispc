
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int8 a = -128, b = 127; // min and max unsigned int8
    RET[programIndex] = saturating_sub(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = -128;
}
