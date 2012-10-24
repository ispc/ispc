// Illegal to provide "varying" allocation count with "uniform new" expression

int * func(int x) {
    return uniform new int[x];
}
