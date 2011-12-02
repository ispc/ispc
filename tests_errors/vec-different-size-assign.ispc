// Can't convert between differently sized vector types

void foo(float<3> a, int<2> b) {
    a += b;
}
