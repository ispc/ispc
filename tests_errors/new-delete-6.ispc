// Can't convert from type "uniform int32 \* varying" to type "uniform int32 \* uniform" for return
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64

int * uniform func(int x) {
    return new int[x];
}
