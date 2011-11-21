
export uniform int width() { return programCount; }

struct Foo {
    int a;
    uniform float b;
};

void update(Foo * varying fp) {
    ++fp;
    fp->a -= 1;
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
