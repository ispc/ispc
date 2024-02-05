// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s


float bar(float a, float b);

export uniform int foo(uniform int x[], uniform int i[]) {
    float (*fptr)(float, float) = bar;
//CO    bar(0,1,2);
// CHECK: Error: Too many parameter values provided in function call
    fptr(0., 1, 2);
// CHECK: Error: Can't convert argument of type "void * uniform" to type "varying float" for function call argument.
    fptr(NULL, 1);
// CHECK: Error: Too few parameter values provided in function call (1 provided, 2 expected).
    fptr(1.);
}
