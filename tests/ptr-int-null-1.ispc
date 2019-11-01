bool foo(int *ptr) {
    return (ptr == NULL);
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = 0;
    RET[programIndex] = foo(0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
