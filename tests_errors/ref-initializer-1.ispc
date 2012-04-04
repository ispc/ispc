// Initializer for reference-type variable "x" must have an lvalue type

float &func(uniform float a[], int i, float f) {
    float &x = 1.; // a[i];
}

