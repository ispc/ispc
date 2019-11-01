export void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 v = 0xffffffff;
    RET[programIndex] = (v == 0xffffffffll) ? 1 : 0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
