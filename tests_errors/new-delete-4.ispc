// Type conversion only possible from atomic types

struct P { int x; };

int * func(P p) {
    return new int[p];
}
