// Can't compute the size of declared but not defined struct type

struct Foo;

uniform int bar() {
    return sizeof(Foo);
}
