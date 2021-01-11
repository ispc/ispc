// syntax error, unexpected '\('
// rule: skip on arch=genx32
// rule: skip on arch=genx64

int * func(int a) {
    return new int[a](10);
}
