
export uniform int width() { return programCount; }

void foo(uniform float * uniform * ret) {
    uniform float *px = *ret;
    ++px;
    *ret = px;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ptr = &aFOO[0];
    foo(&ptr);
    RET[programIndex] = *ptr;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
