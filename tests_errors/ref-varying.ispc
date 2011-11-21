// fff

void foo(float &x) {
    ++x;
}

void bar(uniform float a[], int i) {
    foo(a[i]);
}
