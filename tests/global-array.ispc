
uniform float a[100];

export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = i;
    RET[programIndex] = a[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
