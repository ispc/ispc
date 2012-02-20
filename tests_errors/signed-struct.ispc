// "signed" qualifier is illegal with non-integer type "varying struct Foo"

struct Foo {
    float x;
};

signed Foo f;
