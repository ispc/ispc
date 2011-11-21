// Inconsistent expression list lengths found in initializer list

void foo() {
    int a[2][] = { { 1, 2, 3 }, { 1, 2, 3, 4 } };
}

