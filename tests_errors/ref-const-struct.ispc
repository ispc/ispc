// Can't assign to type "const varying int32" on left-hand side of expression

struct Foo {
    int x;
};

void f(const Foo &f) {
    ++f.x;
}
