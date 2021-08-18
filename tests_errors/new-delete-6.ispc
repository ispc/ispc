// Can't convert from type "uniform int32 \* varying" to type "uniform int32 \* uniform" for return
// rule: skip on arch=xe32
// rule: skip on arch=xe64

int * uniform func(int x) {
    return new int[x];
}
