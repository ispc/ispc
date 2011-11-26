// Illegal to assign to type "uniform struct Foo" due to element "a" with type "const int32"

struct Foo {
    const int a;
};

void foo(Foo f) {
    Foo a;
    a = f;
}
