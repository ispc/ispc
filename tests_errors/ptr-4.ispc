// Illegal to perform pointer arithmetic

void *foo(void *p) {
    p += 1;
    return p;
}
