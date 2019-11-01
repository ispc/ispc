struct Foo {
    int a;
    uniform float b;
};

void update(varying Foo * uniform fp) {
    fp->a += 1;
    fp->b = 1;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo f = { aFOO[programIndex], 5 };
    update(&f);
    RET[programIndex] = f.b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
