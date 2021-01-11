// syntax error, unexpected 'const'.
// rule: skip on arch=genx32
// rule: skip on arch=genx64

int * func(int a) {
    return const new int[a];
}
