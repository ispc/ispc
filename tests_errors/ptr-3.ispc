// Illegal to perform pointer arithmetic

void *foo(void *p) {
    p = p-2;
    return p;
}
