struct Foo {
    int a;
    uniform float b;
};

void update(varying Foo * varying fp) {
    ++fp;
    #pragma ignore warning(perf)
    fp->a -= 1;
    // Revisit: this should be no different from writing to any uniform
    // variable. So, writing to same location warning might be not needed
    #pragma ignore warning
    fp->b = 1;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo f[2] = { { 1234, 4321 }, { aFOO[programIndex], 5 } };
    update(f);
    RET[programIndex] = f[1].a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
