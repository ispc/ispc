// Illegal to delete non-pointer type
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64
void func(int a) {
    delete a;
}
