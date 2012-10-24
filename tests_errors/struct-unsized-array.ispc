// Unsized arrays aren't allowed except for the last member in a struct definition.

struct Foo {
    float a[];
    float b[10];
};

Foo f;

void Func() {
    Foo f;
}

