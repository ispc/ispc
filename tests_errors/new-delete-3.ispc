// syntax error, unexpected '\('
// rule: run on arch=x86
// rule: run on arch=x86-64
int * func(int a) {
    return new int[a](10);
}
