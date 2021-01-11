// Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible
// rule: skip on arch=genx32
// rule: skip on arch=genx64

struct P { int x; };

int * func(P p) {
    return new int[p];
}
