// Can't convert between different struct types "uniform struct Foo" and "uniform struct Bar"

struct Foo {
    int a; 
};

struct Bar {
    int x; 
};

void f(Foo a) { 
    Bar b;
    b  =a;
}
