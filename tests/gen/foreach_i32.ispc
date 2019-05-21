export void vecAdd(uniform int * uniform A, uniform int * uniform B, uniform int * uniform C, uniform int size) {
  foreach(i = 0 ... size) {
      C[i] = A[i] + B[i];
  }
}