// Inconsistent initializer expression list lengths make it impossible to size unsized array dimensions

void foo() {
    int a[2][] = { { 1, 2, 3 }, { 1, 2, 3, 4 } };
}

