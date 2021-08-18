// Can't dynamically allocate storage for declared but not defined type
// rule: skip on arch=xe32
// rule: skip on arch=xe64

struct Foo;

Foo * uniform bar() {
    return uniform new Foo;
}
