export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    if (v >= 0 && v < 2)
        m = reduce_min(-(int)v);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { RET[programIndex] = -1; }
