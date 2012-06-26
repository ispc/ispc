// Error: "break" statement is illegal outside of for/while/do loops

void foo(int x) {
    foreach_unique (u in x) {
        if (u == 0)
            break;
    }
}


