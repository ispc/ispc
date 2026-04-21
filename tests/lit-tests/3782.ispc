// Verify that frontend emission uses poison (not undef) for intermediate
// vectors/aggregates that are immediately filled element-wise.
// See https://github.com/ispc/ispc/issues/3782

// RUN: %{ispc} %s --target=host --nowrap --nostdlib --debug-phase=first:first --emit-llvm-text -o - | FileCheck %s

// BroadcastValue: insertelement + shufflevector with poison base
// CHECK-LABEL: @broadcast_test
// CHECK: insertelement <[[WIDTH:[0-9]+]] x i32> poison, i32 %{{.*}}, i32 0
// CHECK-NEXT: shufflevector <[[WIDTH]] x i32> %{{.*}}, <[[WIDTH]] x i32> poison, <[[WIDTH]] x i32> zeroinitializer
void broadcast_test(uniform int val, varying int *uniform out) {
    *out = val;
}

// BinaryOperator array path: insertvalue with poison base
// CHECK-LABEL: @binop_sv
// CHECK: insertvalue [3 x <{{[0-9]+}} x i32>] poison, <{{[0-9]+}} x i32> %{{.*}}, 0
void binop_sv(varying int<3> a, varying int<3> b, varying int<3> *uniform out) {
    *out = a + b;
}

// CmpInst array path + bool lowering: both use poison base
// CHECK-LABEL: @cmp_sv
// CHECK: insertvalue [3 x <[[WIDTH3:[0-9]+]] x i1>] poison, <[[WIDTH3]] x i1> %{{.*}}, 0
// CHECK: insertvalue [3 x <[[WIDTH3]] x i[[BOOLBITS:(1|32)]]>] poison, <[[WIDTH3]] x i[[BOOLBITS]]> %{{.*}}, 0
void cmp_sv(varying int<3> a, varying int<3> b, varying bool<3> *uniform out) {
    *out = a > b;
}

// NotOperator array path: insertvalue with poison base
// CHECK-LABEL: @not_sv
// CHECK: insertvalue [3 x <{{[0-9]+}} x i32>] poison, <{{[0-9]+}} x i32> %{{.*}}, 0
void not_sv(varying int<3> a, varying int<3> *uniform out) {
    *out = ~a;
}