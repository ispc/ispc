// rule: run on arch=x86
// rule: run on arch=x86-64
// syntax error, unexpected 'const'.

int * func(int a) {
    return const new int[a];
}
