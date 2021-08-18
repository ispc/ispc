// Illegal to provide "varying" allocation count with "uniform new" expression
// rule: skip on arch=xe32
// rule: skip on arch=xe64

int * func(int x) {
    return uniform new int[x];
}
