// Illegal to return non-lvalue from function returning reference type

float &func(uniform float a[], int i, float f) {
    return 1.f;
}
