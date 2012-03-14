// Illegal to dereference non-pointer or reference type "varying float"

float func(float a) {
    *a = 0;
    return 0;
}
