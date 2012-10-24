// Can't assign to type

struct Foo {
    int x;
};

int func(const Foo f) {
    f.x = 0;
}
