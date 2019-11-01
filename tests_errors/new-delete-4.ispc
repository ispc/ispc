// rule: run on arch=x86
// rule: run on arch=x86-64
// Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible

struct P { int x; };

int * func(P p) {
    return new int[p];
}
