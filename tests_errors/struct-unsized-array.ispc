// Unsized arrays aren't allowed in struct definitions

struct Foo {
    float a[];
    float b[10];
};

Foo f;

void Func() {
    Foo f;
}

