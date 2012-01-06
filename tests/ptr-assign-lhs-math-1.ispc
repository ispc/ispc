
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount];
    a[programIndex] = aFOO[programIndex];

    uniform float * uniform ptr = a;
    *(ptr+1) = 0;
    RET[programIndex] = a[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
    RET[1] = 0;
}
