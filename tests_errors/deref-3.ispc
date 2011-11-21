// Dereference operator "->" can't be applied to non-pointer type "uniform struct Foo"

struct Foo { int x; };

int func(Foo a) {
    a->x = 0;
}
