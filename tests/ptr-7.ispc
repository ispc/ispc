
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying ap = &aFOO[programIndex];
    RET[programIndex] = *ap;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
