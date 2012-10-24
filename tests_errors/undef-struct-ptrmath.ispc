// Illegal to pre/post increment pointer to undefined struct type

struct Foo;

Foo * uniform bar(Foo * uniform f) {
    return ++f;
}
