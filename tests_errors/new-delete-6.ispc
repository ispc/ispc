// Can't convert from varying type "int32 *" to uniform type "int32 * uniform" for return

int * uniform func(int x) {
    return new int[x];
}
