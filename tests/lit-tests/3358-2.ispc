// RUN: %{ispc} -O2 %s --target=sse4 --arch="x86_64" --emit-llvm-text -o - | FileCheck %s

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

__attribute__((external_only)) export void foo(uniform int64 Result[], uniform int64 Source[]) {
  uniform S R;
  foreach(i = 0 ... 2)
      ((uniform int64 *)&R)[i] = Source[i];
  *((uniform S *)Result) = R;
}
