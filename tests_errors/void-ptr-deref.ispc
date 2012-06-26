// Illegal to dereference void pointer type

int foo(void * uniform ptr) {
    print("%", *ptr);
}
