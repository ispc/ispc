export void vecAdd(uniform float * uniform A, uniform float * uniform B, uniform float * uniform C, uniform int size) {
    int index = programIndex;
	float v = A[index];
	if (v > 5.0f) {
	  for (uniform int j=0; j< size; j+=programCount) {
		C[programIndex+j] = A[programIndex+j] + B[programIndex+j];
	  }
	}
}
