export void vecAdd(uniform float * uniform A, uniform float * uniform B, uniform float * uniform C, uniform int size) {
  foreach(i = 0 ... size) {
      C[i] = A[i] + B[i];
  }
}