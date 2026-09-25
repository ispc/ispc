// RUN: %{ispc} --pic --target=neon-i32x4 %s --emit-asm -o - | FileCheck %s

// REQUIRES: ARM_ENABLED

// CHECK-LABEL: RoundToInt___vyf:{{.*}}
// CHECK: frintn{{.*4s.*}}
// CHECK-NEXT: ret
float RoundToInt(float x)
{
    return round(x);
}

// CHECK-LABEL: RoundToInt___vyd:{{.*}}
// CHECK: frintn{{.*2d.*}}
// CHECK-NEXT: frintn{{.*2d.*}}
// CHECK-NEXT: ret
double RoundToInt(double x)
{
    return round(x);
}
