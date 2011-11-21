// operator "." can't be applied to pointer type

struct Foo { int x; };

int func(Foo *a) {
    a.x = 0;
}
