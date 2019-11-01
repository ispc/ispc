struct Foo;

void bing(Foo * uniform);

struct Foo {
    int i;
    varying float f;
    Foo * uniform next;
};

void bar(Foo * uniform f) {
    bing(f);
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform Foo fa, fb;
    fa.next = &fb;
    fb.f = aFOO[programIndex]; 
    fb.i = 100;
    bar(&fa);
    RET[programIndex] = fb.f; 
}


void bing(Foo * uniform f) {
    f = f->next;
    f->f *= 2;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
