// Can't assign to type "const uniform int32"

void foo(int x) {
    foreach_unique (u in x)
        ++u;
}


