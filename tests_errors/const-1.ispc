// Can't assign to type "const varying int32" on left-hand

int func() {
    const int x = 2;
    ++x;
}
