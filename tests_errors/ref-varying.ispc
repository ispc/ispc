// Illegal to pass a "varying" lvalue

void foo(float &x) {
    ++x;
}

void bar(uniform float a[], int i) {
    foo(a[i]);
}
