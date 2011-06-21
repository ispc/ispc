
struct Foo {
    float f;
};


export float foo(Foo f[], int i, uniform int j) {
    Foo x = f[i];
    return x.f;
}

