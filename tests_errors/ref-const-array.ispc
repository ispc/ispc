// Can't assign to type "const varying int32" on left-hand side of expression

const int x[20];

void foo() {
    ++x[5];
}
