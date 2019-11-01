export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    typedef float Float[3];
    float a = aFOO[programIndex];
    Float x;
    x[0] = 1;
    x[1] = a;
    x[2] = 2;
    RET[programIndex] = x[b-4];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
