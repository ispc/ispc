struct Foo {
    float x, y;
};

struct Bar {
    uniform Foo uf;
    varying Foo vf;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    //    Bar bar = { { b, b }, { a, a } };
    Bar bar;
    bar.uf.x = b;
    bar.vf.y = a;
    RET[programIndex] = bar.uf.x + bar.vf.y;
}

export void result(uniform float RET[]) { RET[programIndex] = 6+programIndex; }
