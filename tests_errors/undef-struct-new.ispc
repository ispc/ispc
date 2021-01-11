// Can't dynamically allocate storage for declared but not defined type
// rule: skip on arch=genx32
// rule: skip on arch=genx64

struct Foo;

Foo * uniform bar() {
    return uniform new Foo;
}
