export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex];
    float y;
    y = ++x, 3;
    RET[programIndex] = y;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2+programIndex;
}
