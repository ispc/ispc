// Illegal to delete non-pointer type
// rule: run on arch=x86
// rule: run on arch=x86-64
void func(int a) {
    delete a;
}
