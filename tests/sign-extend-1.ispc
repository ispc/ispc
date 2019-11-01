export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    bool b = (a == 1.);
    int32 s = sign_extend(b);
    RET[programIndex] = (s == 0xffffffff) ? 16 : 2;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
    RET[0] = 16;
}
