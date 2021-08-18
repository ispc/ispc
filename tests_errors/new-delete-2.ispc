// syntax error, unexpected 'const'.
// rule: skip on arch=xe32
// rule: skip on arch=xe64

int * func(int a) {
    return const new int[a];
}
