// RUN: not %{ispc} %s --target=host --nostdlib --nowrap 2>&1 | FileCheck %s

// CHECK: Error: Trying to index into non-array, vector, or pointer type "uniform float".
// CHECK: Error: Trying to index into non-array, vector, or pointer type "varying float".
// CHECK: Error: Trying to index into non-array, vector, or pointer type "varying float".

// Case 1
export void f(uniform float op,  uniform float x[]) {
    int i,j,k,nlat,nlon;
    op[i*nlat*nlon + j*nlon + k] * x[j*nlon + k];
}

// Case 2
void test(uniform float data[]) {
    varying float foo, bar;
    foreach(k=0...8)
        data[k] = foo[k]*bar[k];
}
