// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED && !ARM_ENABLED

uniform bool<4> x;

// CHECK-LABEL: @foo___unT_3C_4_3E_unT_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ult <4 x i8> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = sext <4 x i1> [[CMP]] to <4 x i8>
// CHECK-NEXT: store <4 x i8> [[CAST]], {{.*}} @x
// CHECK-NEXT: ret void

void foo(uniform uint8<4> a, uniform uint8<4> b)
{
    x = a < b;
}

// CHECK-LABEL: @boo___unb_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CAST:%.*]] = sext <4 x i1> %a to <4 x i8>
// CHECK-NEXT: store <4 x i8> [[CAST]], {{.*}} @x
// CHECK-NEXT: ret void

void boo(uniform bool<4> a) {
    x = a;
}
