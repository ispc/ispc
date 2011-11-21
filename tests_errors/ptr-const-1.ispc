// Can't assign to type "const int32 * const"

void foo(const int * const p) {
    ++p;
}
