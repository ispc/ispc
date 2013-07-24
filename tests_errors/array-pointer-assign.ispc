// Illegal to assign to array type "varying float\[5\]"

void foo(float *x) {
    float a[5] = { 1,2,3,4,5};
    a = x;
}
