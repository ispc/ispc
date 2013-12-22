
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int16 a = 65535; // max unsigned int16
    RET[programIndex] = saturating_add(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (uniform unsigned int16) 65535;
}
