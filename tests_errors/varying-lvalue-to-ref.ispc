// ffofoof

void inc(float &x) { ++x; }

void foo(uniform float a[], int index) {
    inc(a[index]);
}
