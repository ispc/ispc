export void vecAdd(uniform float * uniform A, uniform float * uniform B, uniform float * uniform C, uniform int size) {
  foreach(i = 0 ... M) {
    float sum = 0.0f;
    if (A[i] > 25.0f) {
        for (uniform int j = 0; j<N; j++) {
          sum +=A[i]+ j + B[i];
        }
    }
    C[i] = sum;
}
