export void vecAdd(uniform int * uniform A, uniform int * uniform B, uniform int * uniform C, uniform int size) {
  foreach(i = 0 ... M) {
    int sum = 0;
    if (A[i] > 25) {
        for (uniform int j = 0; j<N; j++) {
          sum +=A[i]+ j + B[i];
        }
    }
    C[i] = sum;
}
