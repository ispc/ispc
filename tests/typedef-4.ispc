export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    typedef float Float;
    Float a = aFOO[programIndex];
    RET[programIndex] = a;
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
