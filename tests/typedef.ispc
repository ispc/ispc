
export uniform int width() { return programCount; }



typedef float Float[3];

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Float x;
    x[0] = a;
    x[1] = 1;
    x[2] = 2;
    RET[programIndex] = x[b-5];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
