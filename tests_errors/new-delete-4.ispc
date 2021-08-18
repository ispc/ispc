// Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible
// rule: skip on arch=xe32
// rule: skip on arch=xe64

struct P { int x; };

int * func(P p) {
    return new int[p];
}
