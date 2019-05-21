export void vecAdd(uniform int * uniform A, uniform int * uniform B, uniform int * uniform C, uniform int size) {
  foreach(i = 0 ... size) {
	int sum = 0;
	for (uniform int j = 0; j<size; j++) {
	  sum += A[i]+j + B[i];
	}
	C[i] = sum;
  }
}
