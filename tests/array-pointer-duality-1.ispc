
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount];
    for (unsigned int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];

    RET[programIndex] = *(a + 2);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
