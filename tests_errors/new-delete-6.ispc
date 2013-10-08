// Can't convert from type "uniform int32 \* varying" to type "uniform int32 \* uniform" for return

int * uniform func(int x) {
    return new int[x];
}
