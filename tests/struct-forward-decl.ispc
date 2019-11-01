struct Foo;

void bing(varying Foo * uniform);

struct Foo {
    float f;
    int i;
};

void bar(varying Foo * uniform f) {
    bing(f);
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo f;
    f.f = aFOO[programIndex]; 
    f.i = programIndex;
    bar(&f);
    RET[programIndex] = f.f; 
}


void bing(varying Foo * uniform f) {
    f->f *= 2;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
