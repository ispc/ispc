void inc(int * uniform v) {
    ++*v;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = &aFOO[0];
    b += 3;
    b -= 1;
    RET[programIndex] = *b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
