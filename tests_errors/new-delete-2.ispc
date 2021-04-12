// syntax error, unexpected 'const'.
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64

int * func(int a) {
    return const new int[a];
}
