// Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Foo { uniform int x; };

void a(uniform Foo * varying array, int index) {
    Foo v = array[index];
}
