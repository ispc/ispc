// rule: run on arch=x86
// rule: run on arch=x86-64
// Illegal to delete non-pointer type

void func(int a) {
    delete a;
}
