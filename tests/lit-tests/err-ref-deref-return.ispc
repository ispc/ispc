// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to dereference non-pointer type "uniform float &".

export void simple_reduction(uniform float vin[], uniform int w, uniform float & result)
{
   float sum = 0;
   foreach (i = 0 ... w) {
       sum += vin[i];
   }
   *result = reduce_add(sum);   // << I would expect this to produce a compiler error
}

// CHECK: Error: Illegal to return non-lvalue from function returning reference type

float &func(uniform float a[], int i, float f) {
    return 1.f;
}

// CHECK: Error: Illegal to return varying lvalue type from function returning a reference type

float &func2(uniform float a[], int i, float f) {
    return a[i];
}