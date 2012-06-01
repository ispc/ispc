// Illegal to take address of non-lvalue or function

void foo() {
    int *ptr = &(1+1);
}
