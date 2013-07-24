// Can't convert from pointer type "void \* varying" to incompatible pointer type "uniform int32 \* varying" for return statement

int *foo(void *p) {
    return p;
}
