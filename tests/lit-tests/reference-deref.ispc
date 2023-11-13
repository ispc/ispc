// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to dereference non-pointer type "uniform float &".

export void simple_reduction(uniform float vin[], uniform int w, uniform float & result)
{
   float sum = 0;
   foreach (i = 0 ... w) {
       sum += vin[i];
   }
   *result = reduce_add(sum);   // << I would expect this to produce a compiler error
}
