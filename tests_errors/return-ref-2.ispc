// Illegal to return varying lvalue type from function returning a reference type

float &func(uniform float a[], int i, float f) {
    return a[i];
}
