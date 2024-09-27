// The test checks that variability of template arguments is consistent across nested template calls.
// RUN: %{ispc} %s --ast-dump --target=host --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// Check for "level2" instantiation
// CHECK: (instantiation <uniform float, uniform int32>) {{.*}} [ uniform float(uniform int32 v)] "level2"
// Check for "level1" instantiation
// CHECK: (instantiation <uniform float, uniform int32>) {{.*}} [ uniform float(uniform int32 v)] "level1"
// Make sure no other instantiations occur
// CHECK-NOT: instantiation <uniform float, uniform int32>
// CHECK-NOT: Error: Can't convert

template<typename UVFloat, typename UVInt>
inline UVFloat level2(UVInt v) {
    return v * 1.f;
}

template<typename UVFloat, typename UVInt>
inline UVFloat level1(UVInt v) {
    return level2<UVFloat, UVInt>(v) +
           level2<UVFloat,uniform int>(v);
           level2<uniform float, UVInt>(v);
}

uniform float testI(uniform int i) {
    return level1<uniform float, uniform int>(i);
}
