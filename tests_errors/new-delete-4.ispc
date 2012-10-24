// Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible

struct P { int x; };

int * func(P p) {
    return new int[p];
}
