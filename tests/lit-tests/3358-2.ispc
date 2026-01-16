// RUN: %{ispc} -O2 %s --target=sse4 --arch="x86_64" --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// Verify that oversized vector stores don't get optimized away as UB.
// Casting to int64 with vector width 4 creates <4 x i64> vectors (256-bit)
// which, when stored via blend operations into 128-bit structures, creates UB.

// CHECK:      define void @foo
// CHECK-NEXT: allocas:
// CHECK-NEXT:   [[RES:%.*]] = load <4 x i32>, ptr %Source
// CHECK-NEXT:   store <4 x i32> [[RES]], ptr %Result
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

struct S {
  uniform int32<4> m;
};

unmasked void foo(uniform int64 Result[], uniform int64 Source[]) {
  uniform S R;
  foreach(i = 0 ... 2)
      ((uniform int64 *)&R)[i] = Source[i];
  *((uniform S *)Result) = R;
}

// CHECK:      define void @bar
// CHECK-NEXT: allocas:
// CHECK-NEXT:   %R = alloca
// CHECK:        [[PTR1:%.*]] = getelementptr i8, ptr %Source, i64 %0
// CHECK-NEXT:   [[S:%.*]] = load <1 x i64>, ptr [[PTR1]]
// CHECK-NEXT:   [[PTR2:%.*]] = getelementptr i8, ptr %R, i64 %0
// CHECK-NEXT:   [[VAL:%.*]] = extractelement <1 x i64> [[S]], i64 0
// CHECK-NEXT:   store i64 [[VAL]], ptr [[PTR2]], align 8
// CHECK-NEXT:   [[R:%.*]] = load <4 x i32>, ptr %R
// CHECK-NEXT:   store <4 x i32> [[R]], ptr %Result
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

unmasked void bar(uniform int64 Result[], uniform int64 Source[], uniform int index) {
  uniform S R;
  foreach(i = 0 ... 1)
      ((uniform int64 *)&R)[index + i] = Source[index + i];
  *((uniform S *)Result) = R;
}
