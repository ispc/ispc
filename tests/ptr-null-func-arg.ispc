bool bar(float * x) {
    return (x != 0);
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = bar(NULL);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
