
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying int8 a = -128, b = aFOO[programIndex]; // max signed int8
    RET[programIndex] = saturating_add(a, -b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying int8) -128;
}
