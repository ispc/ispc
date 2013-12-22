
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying unsigned int16 a = 65535, b = aFOO[programIndex]; // max unsigned int16
    RET[programIndex] = saturating_add(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying unsigned int16) 65535;
}
