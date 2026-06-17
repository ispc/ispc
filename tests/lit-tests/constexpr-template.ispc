// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

constexpr uniform int width() { return 3; }

template <int N>
uniform int foo() { return N; }

uniform int bar() { return foo<width()>(); }
