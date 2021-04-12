// Can't dynamically allocate storage for declared but not defined type
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64

struct Foo;

Foo * uniform bar() {
    return uniform new Foo;
}
