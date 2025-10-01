// This test ensures that ispc compiler generates a zext for an unsigned
// loop induction variable. We also check that nsw flag is not incorrectly
// set on binop operating on the induction variable.

// RUN: %{ispc} %s --emit-llvm-text --target=host --no-discard-value-names -o - -O1 --nowrap | FileCheck %s

// CHECK-LABEL: define void @fill(
// CHECK:      for_loop:
// CHECK-NEXT:   %i.019 = phi i32 [ %add_i_load10_, %for_loop ], [ 0, %allocas ]
// CHECK-NEXT:   %shl_i_load4_broadcast5_.elt0 = shl i32 %i.019, 2
// CHECK-NEXT:   %offset_cast.elt0 = zext i32 %shl_i_load4_broadcast5_.elt0 to i64
// CHECK-NEXT:   %ptr = getelementptr i8, ptr %buffer, i64 %offset_cast.elt0
// CHECK-NEXT:   store <8 x i32> splat (i32 7), ptr %ptr, align 4
// CHECK-NEXT:   %add_i_load10_ = add i32 %i.019, 8
// CHECK-NEXT:   %less_i_load_count_load = icmp ult i32 %add_i_load10_, %count
// CHECK-NEXT:   br i1 %less_i_load_count_load, label %for_loop, label %for_exit

export void fill(uint32* uniform buffer, uniform int count) {
    for (uniform uint i = 0; i < count; i += programCount) {
        buffer[i + programIndex] = 7;
    }
}
