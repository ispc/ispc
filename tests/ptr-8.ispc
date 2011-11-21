
export uniform int width() { return programCount; }

void inc(int * uniform v) {
    ++*v;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = aFOO[programIndex];
    int * uniform b = &a;
    if (a <= 2)
        inc(b);
    RET[programIndex] = a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
    RET[0] = 2;
    RET[1] = 3;
}
