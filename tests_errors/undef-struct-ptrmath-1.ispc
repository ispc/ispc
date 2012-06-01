// Illegal to perform pointer arithmetic on undefined struct type

struct Foo;

Foo * uniform bar(Foo * uniform f) {
    return f + 1;
}
