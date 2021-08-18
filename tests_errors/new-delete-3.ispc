// syntax error, unexpected '\('
// rule: skip on arch=xe32
// rule: skip on arch=xe64

int * func(int a) {
    return new int[a](10);
}
