// Illegal to assign to type "varying struct Bar" in type "varying struct Foo" due to element "a" with type "const varying int32"

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
