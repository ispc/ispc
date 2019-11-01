export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex];
    float a;
    a = (x *= 2.);
    RET[programIndex] = a + x;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 4 * (1+programIndex);
}
