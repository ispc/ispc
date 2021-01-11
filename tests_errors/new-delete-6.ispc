// Can't convert from type "uniform int32 \* varying" to type "uniform int32 \* uniform" for return
// rule: skip on arch=genx32
// rule: skip on arch=genx64

int * uniform func(int x) {
    return new int[x];
}
