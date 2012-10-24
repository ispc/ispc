// Symbol "a" shadows symbol declared in outer scope

void foo() {
    int a;
    {
        int a;
    }
}
