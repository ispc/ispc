// Error: "return" statement is illegal inside a "foreach" loop

void foo(int x) {
    foreach_unique (u in x) {
        if (u == 0)
            return;
    }
}


