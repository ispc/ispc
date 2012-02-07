// Illegal to dereference non-pointer or reference type "float"

float func(float a) {
    *a = 0;
    return 0;
}
