export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m = 42;
    if (v >= 3) {
        #pragma ignore warning(perf)
        m = reduce_min((unsigned int)v);
    }
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? 42 : 3; }
