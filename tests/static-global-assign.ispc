static uniform float x = 2;


export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    if (b > 0)
        x = 0;
    ret[programIndex] = x;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 0;
}
