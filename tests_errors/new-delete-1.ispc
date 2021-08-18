// Illegal to delete non-pointer type
// rule: skip on arch=xe32
// rule: skip on arch=xe64
void func(int a) {
    delete a;
}
