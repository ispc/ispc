
uniform float a[100];

export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = i;
    RET[programIndex] = a[programIndex+b-5]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
