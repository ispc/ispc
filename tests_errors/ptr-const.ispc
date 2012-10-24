// Can't assign to type "const varying int32" on left-hand side

void foo(const int * p) {
    *p = 0;
}
