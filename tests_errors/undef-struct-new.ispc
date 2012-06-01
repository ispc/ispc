// Can't dynamically allocate storage for declared but not defined type

struct Foo;

Foo * uniform bar() {
    return uniform new Foo;
}
