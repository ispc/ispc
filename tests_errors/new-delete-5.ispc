// Illegal to provide "varying" allocation count with "uniform new" expression
// rule: skip on arch=genx32
// rule: skip on arch=genx64

int * func(int x) {
    return uniform new int[x];
}
