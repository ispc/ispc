// This test checks reduced alignment for varying atomic types
// RUN: %{ispc} %s --nostdlib --emit-llvm-text -o - | FileCheck %s
typedef float<3> float3;
// CHECK-COUNT-3: load <{{[0-9]+}} x float>, {{.*}}, align 1

extern float3 TestLoadUnaligned(float3 &b) {
    float3 xyz = b.xyz;
    return xyz;
}
