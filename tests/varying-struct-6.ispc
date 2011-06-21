
export uniform int width() { return programCount; }



struct Foo {
    uniform float x, y;
};

struct Bar {
    uniform Foo uf;
    varying Foo vf;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Bar bar = { { b, b }, { a, a } };
    RET[programIndex] = bar.uf.x + bar.vf.y;
}

export void result(uniform float RET[4]) { RET[programIndex] = 6+programIndex; }
