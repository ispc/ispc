// Can't assign to type "const uniform int32 \* const varying"

void foo(const int * const p) {
    ++p;
}
