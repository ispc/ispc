
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 a = 127; // max signed int8
    RET[programIndex] = saturating_add(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (uniform int8) 127;
}
