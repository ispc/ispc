// Illegal to provide "varying" allocation count with "uniform new" expression
// rule: run on arch=x86
// rule: run on arch=x86-64

int * func(int x) {
    return uniform new int[x];
}
