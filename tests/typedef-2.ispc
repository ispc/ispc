
export uniform int width() { return programCount; }

typedef float FLOAT;

struct Foo {
    FLOAT x[16];
};

typedef Foo FOO;

struct Bar {
    FOO foo[16];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Bar bar = b;
    RET[programIndex] = bar.foo[a-1].x[a-1];
}

export void result(uniform float RET[]) { RET[programIndex] = 5; }
