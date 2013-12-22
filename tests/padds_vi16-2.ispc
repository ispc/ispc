
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int16 a = -32768, b = aFOO[programIndex]; // max signed int16
    RET[programIndex] = saturating_add(a, -b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying int16) -32768;
}
