// Illegal to assign to type "uniform struct Bar" in type "uniform struct Foo" due to element "a" with type "const int32"

struct Bar {
    const int a;
};

struct Foo {
    struct Bar b;
};

void foo(Foo f) {
    Foo g;
    g = f;
}
