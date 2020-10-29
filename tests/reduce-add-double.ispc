// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    double v = aFOO[programIndex];
    uniform float m;
    if (v < 3)
        m = reduce_add(-v);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? -1 : -3; }
