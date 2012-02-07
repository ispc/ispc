
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount+4];
    for (unsigned int i = 0; i < programCount+4; ++i)
        a[i] = aFOO[min((int)i, programCount)];

    RET[programIndex] = *(a + 2);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 2 : 3;
}
