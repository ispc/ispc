//; RUN: %{ispc} %s --target=host -o %t.o --nostdlib

// Check that we don't fail in parser.

struct S {
    float d2;
};

export void foo(uniform S& dst) {
    dst.d2 = 1;
}
