export void vecAdd(uniform int * uniform A, uniform int * uniform B, uniform int * uniform C, uniform int size) {
    int index = programIndex;
	int v = A[index];
	if (v > 5) {
	  for (uniform int j=0; j< size; j+=programCount) {
		C[programIndex+j] = A[programIndex+j] + B[programIndex+j];
	  }
	}
}
