export void vecAdd(uniform float * uniform A, uniform float * uniform B, uniform float * uniform C, uniform int size) {
  foreach(i = 0 ... size) {
    float v = A[i];
    if (v > 3.0f) {
      C[i] = A[i] + B[i];
    }
  }
}
