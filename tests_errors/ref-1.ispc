// syntax error, unexpected '&'

int foo(int & & bar) {
    bar = 0;
    return bar;
}
