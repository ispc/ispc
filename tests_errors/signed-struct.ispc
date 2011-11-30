// "signed" qualifier is illegal with non-integer type "uniform struct Foo"

struct Foo {
    float x;
};

signed Foo f;
