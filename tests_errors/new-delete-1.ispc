// Illegal to delete non-pointer type
// rule: skip on arch=genx32
// rule: skip on arch=genx64
void func(int a) {
    delete a;
}
