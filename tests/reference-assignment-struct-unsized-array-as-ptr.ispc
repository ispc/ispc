
export uniform int width() { return programCount; }



struct Foo {
    uniform float array[];
};

export void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    Foo f;
    f.array = ret;
    f.array[programIndex] = b;
}

export void result(uniform float r[]) {
    r[programIndex] = 5;
}
